import torch
import torch.nn.functional as F
from datasets import Dataset
import random
import sys
import math
import numpy as np

from .process_datasets import get_data_collator
from models.create_model import LoRALayer

def set_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ======================== DisentAFL loss ================================
def compute_orthogonal_loss(all_latents, latent_info, ortho_loss_weight=0.5):
    if len(all_latents) < 2:
        return torch.tensor(0.0, device=all_latents[0].device)
    
    # Group by (modality, task)
    group_dict = {}
    for latent, (mod, task) in zip(all_latents, latent_info):
        key = (mod, task)
        if key not in group_dict:
            group_dict[key] = []
        group_dict[key].append(latent.mean(dim=1))
    # Stack groups
    group_reps = {
        k: torch.cat(v, dim=0) if len(v) > 1 else v[0]
        for k, v in group_dict.items()
    }
    # Convert to list and sort for consistent ordering
    groups = sorted(group_reps.items(), key=lambda x: (x[0][0], x[0][1]))  # Sort by mod then task
    total_loss = 0.0
    count = 0

    in_loop = False
    if len(groups) > 1:
        for i in range(len(groups)):
            (mod_i, task_i), reps_i = groups[i]
            reps_i = F.normalize(reps_i, dim=1)
            
            for j in range(i + 1, len(groups)):
                (mod_j, task_j), reps_j = groups[j]
                reps_j = F.normalize(reps_j, dim=1)
                
                chunk_size = min(16, reps_i.size(0)) 
                sim_chunks = []
                for k in range(0, reps_i.size(0), chunk_size):
                    in_loop = True
                    chunk_i = reps_i[k:k+chunk_size]
                    chunk_sim = torch.mm(chunk_i, reps_j.T).pow(2).mean()
                    sim_chunks.append(chunk_sim)
                total_loss += sum(sim_chunks) / len(sim_chunks)
                count += 1
    del group_dict, group_reps, groups
    if in_loop:
        del reps_i, reps_j, sim_chunks, chunk_i
    torch.cuda.empty_cache()
    return ortho_loss_weight * (total_loss / count) if count > 0 else torch.tensor(0.0, device=all_latents[0].device)

# ======================== Training Utilities ============================
def learning_rate_with_warmup(current_round, total_rounds=200, warmup_rounds=20, initial_lr=3e-4, min_lr=1e-4): 
    """
    Compute the learning rate with warmup.
    Sourced from and edited from: https://github.com/rui-ye/OpenFedLLM/blob/main/utils/utils.py (OpenFedLLM, KKD)
    """
    if current_round < warmup_rounds:
        # Linear warmup
        lr = min_lr + (initial_lr - min_lr) * (current_round / warmup_rounds)
    else:
        lr = initial_lr
    print(f"the learning rate is {lr}")
    return lr

def get_optimizer(model, round_num, getAdjust, min_lr, peak_lr):
    """Optimizer that only updates trainable parameters with warmup"""
    lora_params_named = {p for n, p in model.named_parameters() if "lora" in n}
    lora_params_type = {p for m in model.modules() if isinstance(m, LoRALayer) for p in m.parameters()}
    lora_params_set = lora_params_named | lora_params_type
    regular_params = [p for p in model.parameters() if p not in lora_params_set]
    lora_params = list(lora_params_set)
    lr = learning_rate_with_warmup(current_round=round_num, initial_lr=peak_lr, min_lr=min_lr) \
        if getAdjust else learning_rate_with_warmup(current_round=0, initial_lr=peak_lr, min_lr=min_lr)
    return torch.optim.AdamW(
        [
            {"params": regular_params, "lr": lr},
            {"params": lora_params, "lr": lr},
        ],
        weight_decay=0.01
    )

def knowledge_distillation_loss(
    teacher_logits, student_logits, 
    temperature=1.0 
):
    # Soften logits with temperature
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    
    # KL Divergence loss (scaled by T^2)
    loss_kd = F.kl_div(
        soft_student, soft_teacher, 
        reduction='batchmean', log_target=False
    ) * (temperature ** 2)
    return loss_kd


# ============== training FL =========== #
def train_client(client_model, client_data, client_id, device='cuda', 
                batch_size=32, accumulation_steps=4, local_rounds=20, round_num=0,
                teacher_model=None,
                personalized=True,
                margin=0.01, smaller_margin=0.005,
                teacher_history=None, student_history=None, components=None,
                useDisentAFL=False, tokenizer=None,
                min_lr=1e-4, peak_lr=3e-4): 
    """Train a client model on its data."""
    client_model.to(device)
    client_model.train()
    if teacher_model:
        teacher_model.eval()
        teacher_model.to(device)

    use_swap_batch = False
    if teacher_model and personalized:
        dataset_tasks = list(components[client_id].keys())
        for task in dataset_tasks:
            if 'image_generation' in task:
                if (teacher_history[task] + smaller_margin) < student_history[task]:
                    replaced = client_model.replace_components(teacher_model, task,
                            teacher_history[task], student_history[task], 
                            smaller_margin, components[client_id])
                    
                    if replaced is True:
                        use_swap_batch = True # use a different shuffle of data to prevent memorization
            else:
                used_swapMargin = margin if 'text_classification' not in task \
                                    else smaller_margin

                if (teacher_history[task] + used_swapMargin) < student_history[task]:
                    replaced = client_model.replace_components(teacher_model, task,
                            teacher_history[task], student_history[task],
                            used_swapMargin, components[client_id])
                    
                    if replaced is True:
                        use_swap_batch = True # use a different shuffle of data to prevent memorization
    
    # Prepare datasets
    all_datasets = []
    task_mapping = {}
    domain_mapping = {}
    for _, (dataset, task_name, _, domain_id) in client_data['datasets'].items():
        if isinstance(dataset, Dataset):
            samples = [{k: v for k, v in example.items()} for example in dataset]
            all_datasets.extend(samples)
            task_mapping.update({len(all_datasets)-1-i: task_name for i in range(len(samples))})
            domain_mapping.update({len(all_datasets)-1-i: domain_id for i in range(len(samples))})
        else:
            raise ValueError("Provided dataset is not of HuggingFace format")
    
    # Sample data for this round
    random.seed(round_num * 2 if use_swap_batch else round_num) # use different subset of data if swapped, dont want to train again on same data
    sample_indices = random.sample(range(len(all_datasets)), min(len(all_datasets), batch_size * accumulation_steps * local_rounds))
    samples = [all_datasets[i] for i in sample_indices]
    sample_tasks = [task_mapping[i] for i in sample_indices]
    sample_domains = [domain_mapping[i] for i in sample_indices]
    
    # Create dataset that includes task information
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

    # Create dataloader
    collate_fn = get_data_collator(client_model.modalities)
    sampler = torch.utils.data.RandomSampler(
            task_dataset, 
            replacement=False, 
            generator=torch.Generator().manual_seed(round_num)
    )
    dataloader = torch.utils.data.DataLoader(
        task_dataset,
        batch_size=batch_size,
        sampler=sampler,  
        collate_fn=collate_fn
    )

    optimizer = get_optimizer(client_model, round_num=round_num, getAdjust=True, min_lr=min_lr, peak_lr=peak_lr)
    total_loss = 0
    total_task_only_loss = 0.0
    overall_teacher_loss = {}
    overall_student_loss = {}
    loss_per_task = {}

    set_seeds(round_num)
    for i, batch in enumerate(dataloader):
        batch_tasks = batch['task']
        unique_tasks = list(set(batch_tasks))
        
        inputs = {}
        for task in unique_tasks:
            if 'image' in client_model.modalities:
                inputs['image'] = batch['image']
            if 'text' in client_model.modalities:
                inputs['text'] = {
                    'input_ids': batch['text']['input_ids'],
                    'attention_mask': batch['text']['attention_mask'],
                }
            if 'tabular' in client_model.modalities:
                inputs['tabular'] = {k: v for k, v in batch['tabular'].items()}

        loss = torch.tensor(0.0, requires_grad=True)
        task_only_loss = torch.tensor(0.0).to(device)

        #disentAFL utils
        if useDisentAFL:
            batch_ortho_loss = torch.tensor(0.0).to(device)   
            all_latents = []
            all_latent_info = []
            
        
        random.shuffle(unique_tasks) 
        for task in unique_tasks:

            task_mask = [t == task for t in batch_tasks]
            domain_ids = [id for index, id in enumerate(batch['domain_id']) if task_mask[index]]
            task_weight = sum(task_mask) / batch_size

            task_inputs = {}
            for modality, data in inputs.items():
                if isinstance(data, dict):
                    task_inputs[modality] = {}
                    for k in data:
                        selected = [v for v_idx, v in enumerate(data[k]) if task_mask[v_idx]]
                        if any(v is None for v in selected):
                            continue
                        task_inputs[modality][k] = torch.stack(selected).to(device) if selected else []
                else:
                    selected = [v for v_idx, v in enumerate(data) if task_mask[v_idx]]
                    if any(v is None for v in selected):
                        continue
                    task_inputs[modality] = torch.stack(selected).to(device) if selected else []

            # Forward pass
            if useDisentAFL:
                outputs, (all_disentAFL_latents, latent_info) = client_model(task_inputs, domain_ids[0], task=task, useDisentAFL=True)
                all_latents.extend(all_disentAFL_latents)
                all_latent_info.extend(latent_info)
            else:
                outputs, comp = client_model(task_inputs, domain_ids[0], task=task)
                components[client_id][task] = comp

            # Calculate loss
            if 'classification' in task:
                labels = batch.get('label', batch.get(task, None))
                if labels is not None:
                    labels = [label for label, mask in zip(labels, task_mask) if mask]
                    labels = torch.tensor(labels).to(device) 
                    classification_loss = task_weight * F.cross_entropy(outputs, labels)

                    if not personalized: # regular client, keep track of teacher loss for personal model
                        if task in overall_teacher_loss:
                            overall_teacher_loss[task] += classification_loss.item()
                        else:
                            overall_teacher_loss[task] = classification_loss.item()
                    else:
                        if task in overall_student_loss:
                            overall_student_loss[task] += classification_loss.item()
                        else:
                            overall_student_loss[task] = classification_loss.item()
                    loss = loss + classification_loss
                    task_only_loss = task_only_loss + classification_loss

                    if task in loss_per_task:
                        loss_per_task[task] += classification_loss.item()
                    else:
                        loss_per_task[task] = classification_loss.item()

            elif 'generation' in task:
                if 'image' in task:
                    image_generation_loss = task_weight * F.mse_loss(outputs, task_inputs['image'])
                    
                    if not personalized: # regular client, keep track of teacher loss for personal model
                        if task in overall_teacher_loss:
                            overall_teacher_loss[task] += image_generation_loss.item()
                        else:
                            overall_teacher_loss[task] = image_generation_loss.item()
                    else:
                        if task in overall_student_loss:
                            overall_student_loss[task] += image_generation_loss.item()
                        else:
                            overall_student_loss[task] = image_generation_loss.item()
                    loss = loss + image_generation_loss
                    task_only_loss = task_only_loss + image_generation_loss

                    if task in loss_per_task:
                        loss_per_task[task] += image_generation_loss.item()
                    else:
                        loss_per_task[task] = image_generation_loss.item()
                else:
                    labels = batch.get('label', batch.get(task, None))
                    labels = [label for label, mask in zip(labels, task_mask) if mask]
                    labels = torch.stack([torch.tensor(v) for v in labels]).to(device)
                    text_generation_loss = (task_weight * F.cross_entropy(
                        outputs.view(-1, outputs.size(-1)),
                        labels.view(-1),
                        ignore_index=tokenizer.tokenizer.pad_token_id #tokenizer.pad_token_id
                    ))

                    if not personalized: # regular client, keep track of teacher loss for personal model
                        if task in overall_teacher_loss:
                            overall_teacher_loss[task] += text_generation_loss.item()
                        else:
                            overall_teacher_loss[task] = text_generation_loss.item()
                    else:
                        if task in overall_student_loss:
                            overall_student_loss[task] += text_generation_loss.item()
                        else:
                            overall_student_loss[task] = text_generation_loss.item()
                    loss = loss + text_generation_loss
                    task_only_loss = task_only_loss + text_generation_loss

                    if task in loss_per_task:
                        loss_per_task[task] += text_generation_loss.item()
                    else:
                        loss_per_task[task] = text_generation_loss.item()
            
        # Calculate orthogonal loss
        if useDisentAFL and (len(inputs.keys()) > 1 or len(unique_tasks) > 1):
            chunk_size = 16  # Adjust based on GPU memory, GPU cant handle entire batch at once
            num_chunks = len(all_latents) // chunk_size + 1
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(all_latents))
                chunk_latents = all_latents[start:end]
                chunk_info = all_latent_info[start:end]
                chunk_loss = compute_orthogonal_loss(chunk_latents, chunk_info)
                batch_ortho_loss += chunk_loss 
            batch_ortho_loss = batch_ortho_loss / num_chunks 
            loss = loss + batch_ortho_loss
        
        # Backward pass
        loss = loss / accumulation_steps
        loss.backward()
        total_loss += loss.item() * accumulation_steps
        total_task_only_loss += task_only_loss.item()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    if not personalized:
        for k, v in overall_teacher_loss.items():
            teacher_history[k] = v / len(dataloader) 
    else:
        for k, v in overall_student_loss.items():
            student_history[k] = v / len(dataloader) 

    print(f"\nRound {round_num}, Loss: {total_loss/len(dataloader):.4f}")
    if personalized or useDisentAFL:
        print(f"Round {round_num}, Task Loss: {total_task_only_loss/len(dataloader):.4f}")
    for task, loss_t in loss_per_task.items():
        print(f"Round {round_num} un-normalized loss for specific task {task}: {loss_t}")

    sys.stdout.flush()
    return total_loss / len(dataloader)

# =============== Training with KD or regular after FL ========================
def train_finetune(client_model, client_data, device='cuda', 
                  batch_size=32, accumulation_steps=4,
                  distil_loss_weight=2e-3, teacher_model=None,
                  personalized=False,
                  seed=42, total_iterations=50, useDisentAFL=False,
                  tokenizer=None,
                  min_lr=1e-4, peak_lr=3e-4):
    """Train a client model on its data."""
    original_knowledge_weight = distil_loss_weight
    client_model.to(device)
    client_model.train()
    if teacher_model:
        teacher_model.eval()
        teacher_model.to(device)
    
    # Prepare datasets
    all_datasets = []
    task_mapping = {}
    domain_mapping = {}
    for _, (dataset, task_name, _, domain_id) in client_data['datasets'].items():
        if isinstance(dataset, Dataset):
            samples = [{k: v for k, v in example.items()} for example in dataset]
            all_datasets.extend(samples)
            task_mapping.update({len(all_datasets)-1-i: task_name for i in range(len(samples))})
            domain_mapping.update({len(all_datasets)-1-i: domain_id for i in range(len(samples))})
        else:
            raise ValueError("Provided dataset is not of HuggingFace format")
    
    class TaskDataset(torch.utils.data.Dataset):
        def __init__(self, data, tasks, domains):
            self.data = data
            self.tasks = tasks
            self.domain_ids = domains
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return {**self.data[idx], 'task': self.tasks[idx], 'domain_id': self.domain_ids[idx]}
    
    full_task_dataset = TaskDataset(all_datasets, 
                                  [task_mapping[i] for i in range(len(all_datasets))], 
                                  [domain_mapping[i] for i in range(len(all_datasets))])
    collate_fn = get_data_collator(client_model.modalities)
    sampler = torch.utils.data.RandomSampler(
            full_task_dataset, 
            replacement=False, 
            generator=torch.Generator().manual_seed(seed)
    )
    dataloader = torch.utils.data.DataLoader(
        full_task_dataset,
        batch_size=batch_size,
        sampler=sampler,  
        collate_fn=collate_fn
    )

    optimizer = get_optimizer(client_model, round_num=199, getAdjust=False, min_lr=min_lr, peak_lr=peak_lr)
    
    set_seeds(seed=seed)
    printed_loss = 0.0
    printed_distil_loss = 0.0
    for iteration in range(total_iterations):
        total_loss = 0
        total_task_only_loss = 0.0
        total_distil_loss = 0.0
        
        # Get the next batch (will cycle through dataset due to iterator)
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            # Reset iterator if we reach end of dataset
            dataloader = torch.utils.data.DataLoader(
                full_task_dataset,
                batch_size=batch_size,
                sampler=sampler,  
                collate_fn=collate_fn
            )
            batch = next(iter(dataloader))
        
        batch_tasks = batch['task']
        unique_tasks = list(set(batch_tasks))
        
        inputs = {}
        for task in unique_tasks:
            if 'image' in client_model.modalities:
                inputs['image'] = batch['image']
            if 'text' in client_model.modalities:
                inputs['text'] = {
                    'input_ids': batch['text']['input_ids'],
                    'attention_mask': batch['text']['attention_mask'],
                }
            if 'tabular' in client_model.modalities:
                inputs['tabular'] = {k: v for k, v in batch['tabular'].items()}

        loss = torch.tensor(0.0, requires_grad=True)
        task_only_loss = torch.tensor(0.0).to(device)
        batch_distil_loss = torch.tensor(0.0).to(device)

        if useDisentAFL:
            batch_ortho_loss = torch.tensor(0.0).to(device)   
            all_latents = []
            all_latent_info = []
            
        random.seed(iteration)
        random.shuffle(unique_tasks) 
        for task in unique_tasks:
            # re-weight based off task for knowledge distillation
            distil_loss_weight = original_knowledge_weight
            if 'text_classification' in task:
                distil_loss_weight = original_knowledge_weight * 500

            task_mask = [t == task for t in batch_tasks]
            domain_ids = [id for index, id in enumerate(batch['domain_id']) if task_mask[index]]
            task_weight = sum(task_mask) / batch_size

            task_inputs = {}
            for modality, data in inputs.items():
                if isinstance(data, dict):
                    task_inputs[modality] = {}
                    for k in data:
                        selected = [v for v_idx, v in enumerate(data[k]) if task_mask[v_idx]]
                        if any(v is None for v in selected):
                            continue
                        task_inputs[modality][k] = torch.stack(selected).to(device) if selected else []
                else:
                    selected = [v for v_idx, v in enumerate(data) if task_mask[v_idx]]
                    if any(v is None for v in selected):
                        continue
                    task_inputs[modality] = torch.stack(selected).to(device) if selected else []

            if useDisentAFL:
                outputs, (all_disentAFL_latents, latent_info) = client_model(task_inputs, domain_ids[0], task=task, useDisentAFL=True)
                all_latents.extend(all_disentAFL_latents)
                all_latent_info.extend(latent_info)
            else:
                outputs, _ = client_model(task_inputs, domain_ids[0], task=task)

            distil_loss = 0.0
            # Calculate loss
            if 'classification' in task:
                labels = batch.get('label', batch.get(task, None))
                if labels is not None:
                    labels = [label for label, mask in zip(labels, task_mask) if mask]
                    labels = torch.tensor(labels).to(device) 
                    classification_loss = task_weight * F.cross_entropy(outputs, labels)

                    if teacher_model and personalized:
                        with torch.no_grad():
                            teacher_outputs = teacher_model(task_inputs, domain_ids[0], task=task)
                        
                        distil_loss = knowledge_distillation_loss(teacher_outputs, outputs) 
                        batch_distil_loss = batch_distil_loss + (distil_loss_weight * distil_loss)
                    loss = loss + classification_loss
                    task_only_loss = task_only_loss + classification_loss

            elif 'generation' in task:
                if 'image' in task:
                    image_generation_loss = task_weight * F.mse_loss(outputs, task_inputs['image'])

                    if teacher_model and personalized:        
                        with torch.no_grad():
                            teacher_outputs = teacher_model(task_inputs, domain_ids[0], task=task)
                        
                        distil_loss = knowledge_distillation_loss(teacher_outputs, outputs) 
                        batch_distil_loss = batch_distil_loss + (distil_loss_weight * distil_loss)
                    loss = loss + image_generation_loss
                    task_only_loss = task_only_loss + image_generation_loss
                else:
                    labels = batch.get('label', batch.get(task, None))
                    labels = [label for label, mask in zip(labels, task_mask) if mask]
                    labels = torch.stack([torch.tensor(v) for v in labels]).to(device)
                    text_generation_loss = (task_weight * F.cross_entropy(
                        outputs.view(-1, outputs.size(-1)),
                        labels.view(-1),
                        ignore_index=tokenizer.tokenizer.pad_token_id #tokenizer.pad_token_id
                    ))
                    
                    if teacher_model and personalized:
                        with torch.no_grad():
                            teacher_outputs = teacher_model(task_inputs, domain_ids[0], task=task)
                        
                        distil_loss = knowledge_distillation_loss(teacher_outputs, outputs) 
                        batch_distil_loss = batch_distil_loss + (distil_loss_weight * distil_loss)
                    loss = loss + text_generation_loss
                    task_only_loss = task_only_loss + text_generation_loss
            
        if personalized:
            loss = loss + batch_distil_loss
        elif useDisentAFL and (len(inputs.keys()) > 1 or len(unique_tasks) > 1):
            chunk_size = 16  # Adjust based on GPU memory, GPU cant handle entire batch at once
            num_chunks = len(all_latents) // chunk_size + 1
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(all_latents))
                chunk_latents = all_latents[start:end]
                chunk_info = all_latent_info[start:end]
                chunk_loss = compute_orthogonal_loss(chunk_latents, chunk_info)
                batch_ortho_loss += chunk_loss
            batch_ortho_loss = batch_ortho_loss / num_chunks 
            loss = loss + batch_ortho_loss
        
        # Backward pass
        loss = loss / accumulation_steps
        loss.backward()
        total_loss += loss.item() * accumulation_steps
        total_distil_loss += batch_distil_loss.item()
        total_task_only_loss += task_only_loss.item()

        if personalized:
            printed_distil_loss += total_distil_loss
        if (iteration + 1) % accumulation_steps == 0 or (iteration + 1) == total_iterations: 
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        printed_loss += total_task_only_loss
    printed_loss = printed_loss / total_iterations
    print(f'The loss is: {printed_loss}')
    if personalized:
        printed_distil_loss = printed_distil_loss / total_iterations
        print(f'The distillation loss is: {printed_distil_loss}')
    sys.stdout.flush()
    return total_loss