import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
from collections import defaultdict

from utils.process_datasets import get_data_collator
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')  # Required for METEOR
from bert_score import score as bertscore

def evaluate_client(client_model, client_data, device='cuda', batch_size=16, tokenizer=None):
    client_model.to(device)
    client_model.eval()

    all_datasets, task_mapping, domain_mapping = [], {}, {}
    for _, (dataset, task_name, _, domain_id) in client_data['datasets'].items():
        if isinstance(dataset, Dataset):
            samples = [dict(example) for example in dataset]
            all_datasets.extend(samples)
            task_mapping.update({len(all_datasets)-1-i: task_name for i in range(len(samples))})
            domain_mapping.update({len(all_datasets)-1-i: domain_id for i in range(len(samples))})
        else:
            raise ValueError("Dataset must be a HuggingFace `Dataset`.")

    samples = [all_datasets[i] for i in range(len(all_datasets))]
    sample_tasks = [task_mapping[i] for i in range(len(all_datasets))]
    sample_domains = [domain_mapping[i] for i in range(len(all_datasets))]

    class TaskDataset(torch.utils.data.Dataset):
        def __init__(self, data, tasks, domains):
            self.data = data
            self.tasks = tasks
            self.domain_ids = domains

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return {**self.data[idx], 'task': self.tasks[idx], 'domain_id': self.domain_ids[idx]}

    task_dataset = TaskDataset(samples, sample_tasks, sample_domains)
    collate_fn = get_data_collator(client_model.modalities)
    dataloader = torch.utils.data.DataLoader(task_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    task_metrics = defaultdict(lambda: defaultdict(list))

    for batch in dataloader:
        batch_tasks = batch['task']
        unique_tasks = list(set(batch_tasks))

        inputs = {}
        if 'image' in client_model.modalities:
            inputs['image'] = batch.get('image')
        if 'text' in client_model.modalities:
            inputs['text'] = batch.get('text')

        for task in unique_tasks:
            task_mask = [t == task for t in batch_tasks]
            domain_ids = [id for idx, id in enumerate(batch['domain_id']) if task_mask[idx]]
            task_inputs = {}
            for modality, data in inputs.items():
                if isinstance(data, dict):
                    task_inputs[modality] = {}
                    for k in data:
                        # Collect only relevant entries for the current task
                        selected = [v for v_idx, v in enumerate(data[k]) if task_mask[v_idx]]
                        if any(v is None for v in selected):
                            continue
                        task_inputs[modality][k] = torch.stack(selected).to(device) if selected else []
                else:
                    selected = [v for v_idx, v in enumerate(data) if task_mask[v_idx]]
                    if any(v is None for v in selected):
                        continue
                    task_inputs[modality] = torch.stack(selected).to(device) if selected else []

            with torch.no_grad():
                outputs = client_model(task_inputs, domain_ids[0], task=task)

            if 'classification' in task:
                labels = batch.get('label', batch.get(task, None))
                labels = [label for i, label in enumerate(labels) if task_mask[i]]
                preds = outputs.argmax(dim=1).cpu().tolist()
                acc = accuracy_score(labels, preds)
                task_metrics[task]['accuracy'].append(acc)

            elif 'regression' in task:
                targets = batch.get('target', batch.get(task, None))
                targets = [t for i, t in enumerate(targets) if task_mask[i]]
                preds = outputs.squeeze().cpu().tolist()
                mse = mean_squared_error(targets, preds)
                task_metrics[task]['mse'].append(mse)

            elif 'generation' in task:
                if 'image' in task:
                    target_images = task_inputs['image']
                    mse = F.mse_loss(outputs, target_images).item()
                    task_metrics[task]['image_mse'].append(mse)
                else:
                    labels = batch.get('label', batch.get(task, None))
                    labels = [label for i, label in enumerate(labels) if task_mask[i]]

                    #### Compute METEOR and BERTScore ####
                    token_ids = torch.argmax(outputs, dim=-1)
                    decoded_preds = tokenizer.tokenizer.batch_decode(
                        token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    decoded_labels = tokenizer.tokenizer.batch_decode(labels, skip_special_tokens=True)
                    
                    # Compute BERTScore
                    P, R, F1 = bertscore(decoded_preds, decoded_labels, lang='en', model_type='bert-base-uncased')
                    avg_f1 = F1.mean().item()
                    task_metrics[task]['bertscore'].append(avg_f1)
                    # Calculate METEOR for each (pred, label) pair
                    meteor_scores = []
                    for pred, label in zip(decoded_preds, decoded_labels):
                        pred_tokens = pred.split()
                        label_tokens = label.split()
                        score = meteor_score([label_tokens], pred_tokens)  # METEOR expects reference as a list of tokens
                        meteor_scores.append(score)
                    task_metrics[task]['meteor'].append(sum(meteor_scores) / len(meteor_scores))

                    labels = torch.stack([torch.tensor(l) for l in labels]).to(device)
                    loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1), ignore_index=tokenizer.tokenizer.pad_token_id).item()
                    task_metrics[task]['cross_entropy'].append(loss)
                    task_metrics[task]['perplexity'].append(np.exp(loss))

    # Aggregate metrics
    aggregated_metrics = {}
    for task, metrics in task_metrics.items():
        aggregated_metrics[task] = {k: float(np.mean(v)) for k, v in metrics.items()}

    return aggregated_metrics