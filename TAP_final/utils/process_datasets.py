from datasets import Dataset
from torchvision import transforms
import torch
import sys

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

def process_client_datasets(datasets_dict, tokenizer, model_type):
    """Process datasets to ensure consistent formats and add templates for text tasks."""
    processed_datasets = {}

    
    for client_id, client_data in datasets_dict.items():
        processed_client = {
            'modalities': client_data['modalities'],
            'tasks': client_data['tasks'],
            'fusion': client_data.get('fusion', False),
            'fusion-groups': client_data.get('fusion-groups', None),
            'dataset_names': client_data['dataset_names'],
            'datasets': {}
        }
        
        print(f'Client {client_id} data processing started')
        sys.stdout.flush()
        for task_name, (dataset, task_type, modality_list, domain_id) in client_data['datasets'].items():
            if isinstance(dataset, Dataset):  # HuggingFace dataset
                if task_type in ['text_generation', 'text_answering']:
                    processed = process_text_generation_dataset(dataset, task_name, task_type, tokenizer, model_type=model_type)
                elif 'classification' in task_type:
                    if 'image' in client_data['modalities'] and 'text' not in task_type:
                        processed = process_image_classification_dataset(dataset, task_type)
                    elif 'text' in client_data['modalities'] and 'image' not in task_type:
                        processed = process_text_classification_dataset(dataset, task_type, tokenizer, model_type=model_type)
                else:
                    # Remove all columns except needed ones
                    cols_to_keep = []
                    if 'image' in client_data['modalities']:
                        if "img" in dataset.column_names:
                            dataset = dataset.rename_column("img", "image")
                        cols_to_keep.append('image')
                    if 'text' in client_data['modalities']:
                        cols_to_keep.extend(['input_ids', 'attention_mask'])
                    if 'label' in dataset.features:
                        cols_to_keep.append('label')
                    if 'target' in dataset.features:
                        cols_to_keep.append('target')
                    
                    processed = dataset.remove_columns(
                        [col for col in dataset.column_names if col not in cols_to_keep]
                    )
            else:  
                raise ValueError("Provided dataset is not a HuggingFace Dataset")
            
            processed_client['datasets'][task_name] = (processed, task_type, modality_list, domain_id)
        print(f'Client {client_id} data processing completed')
        sys.stdout.flush()

        processed_datasets[client_id] = processed_client
    
    return processed_datasets

def process_text_generation_dataset(dataset, task_name, task_type, tokenizer, model_type):
    """Process text generation datasets with appropriate templates."""
    templates = {
        'text_generation': lambda x: x['text'],
        'law_answering': lambda x: format_mmlu_prompt(x, "law"),
        'moral_answering': lambda x: format_mmlu_prompt(x, "ethics"),
        'medicine_answering': lambda x: format_mmlu_prompt(x, "medicine"),
        'common_sense_answering': lambda x: format_commongen_prompt(x),
        'text_answering': lambda x: format_vqa_prompt(x),
    }

    templates_answer = {
        'text_generation': "",
        'law_answering': lambda x: format_mmlu_answer(x, "law"),
        'moral_answering': lambda x: format_mmlu_answer(x, "ethics"),
        'medicine_answering': lambda x: format_mmlu_answer(x, "medicine"),
        'common_sense_answering': lambda x: format_commongen_answer(x),
        'text_answering': lambda x: format_vqa_answer(x),
    }
    
    template_fn = templates.get(task_name, templates['text_generation'])
    template_answer_fn = templates_answer.get(task_name, templates_answer['text_generation'])

    def preprocess(example):
        text = template_fn(example)
        target = template_answer_fn(example)
        
        # Tokenize input (encoder side)
        if model_type == 'ViLT':
            inputs = tokenizer.tokenizer(text, 
                                  padding='max_length', 
                                  truncation=True, max_length=40,
                                  return_tensors="pt"
                                )
        else:
            inputs = tokenizer(
                text=text, 
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )
        
        # Tokenize target (decoder side)
        if model_type == 'ViLT':
            targets = tokenizer.tokenizer(target, 
                                  padding='max_length', 
                                  truncation=True, max_length=40,
                                  return_tensors="pt"
                                )
        else: 
            targets = tokenizer(
                text=target, # target,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )

        result = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': targets['input_ids'].squeeze(0) if isinstance(targets['input_ids'], torch.Tensor) \
                else torch.tensor(targets['input_ids']).squeeze(0) 
        }

        if 'image' in example:
            result['image'] = example['image']
        
        return result
        
    return dataset.map(preprocess, batched=False)

def format_mmlu_prompt(example, subject):
    """Format MMLU questions with choices."""
    choices = '\n'.join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(example['choices'])])
    return (
        f"The following is a multiple-choice question about {subject}. "
        f"Choose the correct answer from the options below.\n\n"
        f"Question: {example['question']}\n"
        f"Options:\n{choices}\n\n"
    )

def format_mmlu_answer(example, subject):
    """Format MMLU output."""
    return (
        f"Correct answer on {subject}: {chr(65 + example['answer'])}. {example['choices'][example['answer']]}"
    )

def format_vqa_prompt(example):
    """Format VQA questions with answers."""
    return (
        f"Answer the question based on the image.\n\n"
        f"Question: {example['question']}\n"
    )

def format_vqa_answer(example):
    """Format VQA questions with answers."""
    if 'multiple_choice_answer' in example:
        answer = example['multiple_choice_answer']
    elif 'answer' in example:
        answer = example['answer']
    else:
        answer = "No answer provided"
    
    return (
        f"Answer: {answer}"
    )

def format_commongen_prompt(example):
    """Format CommonGen questions with choices."""
    choices = '\n'.join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(example['concepts'])])
    return (
        f"Generate an appropriate description from the concepts below.\n\n"
        f"Concepts:\n{choices}\n\n"
    )

def format_commongen_answer(example):
    """Format code generation."""
    answer = example['target']
    return (
        f"Description: {answer}"
    )

def process_text_classification_dataset(dataset, task_type, tokenizer, model_type):
    """Process text classification datasets."""

    def preprocess(batch):
        title = batch.get('title', None)
        description = batch.get('description', None)

        if isinstance(title, str) and isinstance(description, str): #AG News format
            text = f"{title}\n{description}"
            label = torch.tensor(batch['label'] - 1, dtype=torch.long)  # Shift label, ag_news is 1,2,3,4 not 0,1,2,3
        else: 
            text = batch.get('text', '')  # Fallback to 'text' column if available
            label = torch.tensor(batch['label'], dtype=torch.long)  # Keep label as-is

        if model_type == 'ViLT':
            inputs = tokenizer.tokenizer(
                        text, 
                        padding='max_length', 
                        truncation=True, max_length=40,
                        return_tensors="pt"
                    )
        else:
            inputs = tokenizer(
                text=text,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': label
        }

    return dataset.map(preprocess, batched=False)

def process_image_classification_dataset(dataset, task_type):
    """Process image classification datasets."""
    def preprocess(batch):
        # Try to get image data from either key
        image_data = batch.get('image', batch.get('img', None))
        label_data = batch.get('label', batch.get('fine_label', None))

        if image_data is None:
            raise ValueError("Batch must contain either 'image' or 'img' key")
        elif label_data is None:
            raise ValueError("Batch must contain either 'label' or 'fine_label' for cifar100")

        return {
            'image': image_data,
            'label': torch.tensor(label_data, dtype=torch.long)
        }
    return dataset.map(preprocess, batched=False)

def get_data_collator(client_modalities):
    """Get appropriate data collator based on modalities and task."""
    def multi_modal_collator(batch):
        output = {
            'image': [],
            'text': {'input_ids': [], 'attention_mask': []},
            'tabular': {},
            'task': [],
            'domain_id': []
        }

        # Collect labels/targets
        label_keys = ['label', 'target']
        labels = {}
        

        for item in batch:
            if 'task' in item:
                output['task'].append(item['task'])
            if 'domain_id' in item:
                output['domain_id'].append(item['domain_id'])
            if 'image' in client_modalities:
                if 'image' in item:
                    output['image'].append(item['image'])
                elif 'img' in item:
                    output['image'].append(item['img'])
                else:
                    output['image'].append(None) #mask
            if 'text' in client_modalities:
                input_ids = item.get('input_ids')
                attention_mask = item.get('attention_mask')
                if not isinstance(input_ids, torch.Tensor) and input_ids is not None:
                    input_ids = torch.tensor(input_ids)
                    output['text']['input_ids'].append(input_ids)
                else:
                    output['text']['input_ids'].append(None) #mask
                if not isinstance(attention_mask, torch.Tensor) and attention_mask is not None:
                    attention_mask = torch.tensor(attention_mask)
                    output['text']['attention_mask'].append(attention_mask)
                else:
                    output['text']['attention_mask'].append(None) #mask


            # Collect labels
            for key in label_keys:
                if key not in labels:
                    labels[key] = []
                if key in item:
                    labels[key].append(item[key])
                else:
                    labels[key].append(-1)  # Use a default or masking value
                

        if 'image' in client_modalities and output['image']:
            output['image'] = [transform(img) if not isinstance(img, torch.Tensor) and img is not None else img for img in output['image']]

        # Add labels to output
        for key, values in labels.items():
            output[key] = values
        return output
    
    return multi_modal_collator