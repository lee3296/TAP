import argparse
import random
import numpy as np
import torch
import json
from tqdm import tqdm
import os
from collections import defaultdict
from transformers import FlavaProcessor, ViltProcessor

from models.create_model import MultiModalClientModel
from evaluation.load_test_data import load_test_datasets
from evaluation.eval_utils import evaluate_client
from utils.process_datasets import process_client_datasets

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

def initialize_clients(client_datasets, classification_dict, tokenizer, model_type):
    """Initialize client models based on their configurations"""
    clients = {}
    all_task_configs = {}
    dataset_sizes = defaultdict(int)
    for client_id, config in client_datasets.items():
        # Create task configuration for this client
        tasks_config = {}
        
        # Handle different task types
        for _, task in enumerate(config['tasks']):
            if task in ['agnews_text_classification', 
                        'cifar100_image_classification',
                        'imagenet_image_classification']:
                tasks_config[task] = {
                    'type': 'classification',
                    'num_classes': classification_dict[config['dataset_names'][task]]
                }
            elif task == 'image_generation':
                tasks_config[task] = {
                    'type': 'generation',
                    'modality': 'image',
                    'img_size': 64,
                    'channels': 3
                }
            elif task == 'text_generation':
                tasks_config[task] = {
                    'type': 'generation',
                    'modality': 'text',
                    'vocab_size': tokenizer.tokenizer.vocab_size,
                    'max_length': 256 if model_type.lower() == 'flava' else 40
                }
            else:
                raise ValueError(f"Invalid task of {task} has been inputted.")
        
        #get size of local dataset
        for dataset in config['datasets'].values():
            dataset_sizes[client_id] += len(dataset[0])
        
        # Initialize client model
        all_task_configs[client_id] = tasks_config
        clients[client_id] = {
            'datasets': config['datasets'],
            'modalities': config['modalities'],
            'domain_ids': [domain_id for _, (_, _, _, domain_id) in config['datasets'].items()],
            'tasks': config['tasks'],
            'task_domain_pair': [(task, modality_list, domain_id) for _, (_, task, modality_list, domain_id) in config['datasets'].items()],
            'fusion': config['fusion']
        }
        print(f'Client {client_id} has been initalized.')
    return clients, all_task_configs, dataset_sizes

def run_evaluation(args):
    # Set up logging
    os.makedirs(args.save_dir, exist_ok=True)
    if args.training_algo != 'local' and ('main_train' in args.log_dir or 'twotimes' in args.log_dir):
        os.makedirs(os.path.join(args.save_dir, 'regular_clients'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'regularAndLocal_clients'), exist_ok=True)
    if args.training_algo == 'TAP':
        os.makedirs(os.path.join(args.save_dir, 'personal_clients'), exist_ok=True)
    if args.training_algo == 'local':
        os.makedirs(os.path.join(args.save_dir, 'local_clients'), exist_ok=True)
    # Set random seeds
    set_seeds(args.seed)

    tokenizer = FlavaProcessor.from_pretrained("facebook/flava-full")
    if args.model_type == 'ViLT':
        tokenizer = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

    # Load data
    client_datasets = process_client_datasets(load_test_datasets(model_type=args.model_type), tokenizer=tokenizer, model_type=args.model_type)
    
    # Get all modalities and tasks across clients
    all_modalities = set()
    all_tasks = set()
    all_domains = set()
    for config in client_datasets.values():
        all_modalities.update(config['modalities'])
        all_tasks.update(config['tasks'])
        all_domains.update([domain_id for _, (_, _, _, domain_id) in config['datasets'].items()])
    all_modalities = list(all_modalities)
    all_tasks = list(all_tasks)
    all_domains = list(all_domains)
    
    # Register clients with server
    classification_dict = {"sh0416/ag_news": 4,
                           "uoft-cs/cifar100": 100, 'zh-plus/tiny-imagenet': 200}
    clients, all_task_configs, _ = initialize_clients(client_datasets, classification_dict, tokenizer, args.model_type)
    torch.cuda.empty_cache()


    # init personal models and save them on disk, load them in when needed to prevent excessive # of models on GPU and CPU
    group_idx = 0
    for client_id in tqdm(range(args.num_clients), desc="Evaluating clients"):
        group_idx = 0 if group_idx % 10 == 0 else group_idx
        client = clients[group_idx]

        training_client_config = client_datasets[group_idx]
        client_model = MultiModalClientModel(
            modalities=training_client_config['modalities'],
            tasks=all_task_configs[group_idx],
            embed_dim=args.embed_dim,
            fusion=training_client_config['fusion'],
            transformer_option=args.transformer_option,
            domain_ids = [domain_id for _, (_, _, _, domain_id) in training_client_config['datasets'].items()],
            task_domain_pair=[(task, modality_list, domain_id) for _, (_, task, modality_list, domain_id) in training_client_config['datasets'].items()],
            model_type=args.model_type,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            vocab_size=tokenizer.tokenizer.vocab_size, #tokenizer.vocab_size
        ).to(args.device)

        if args.training_algo != 'local' and 'main_train' in args.log_dir:
            client_model.load_state_dict(torch.load(f'{args.log_dir}/regular_clients/finalClient{client_id}.pth'))
        elif args.training_algo != 'local' and 'twotimes' in args.log_dir:
            client_model.load_state_dict(torch.load(f'{args.log_dir}/regularAndLocal_clients/regAndLocalClient{client_id}.pth'))
        elif args.training_algo == 'local':
            client_model.load_state_dict(torch.load(f'{args.log_dir}/local_clients/localClient{client_id}.pth'))
        elif (args.training_algo == 'TAP') and 'main_train' not in args.log_dir: #other exp other than the main one, which also saves two baselines
            client_model.load_state_dict(torch.load(f'{args.log_dir}/personal_clients_postrain/personalClient{client_id}.pth'))
        
        metrics = evaluate_client(
            client_model=client_model,
            client_data=client,
            device=args.device,
            batch_size=args.batch_size,
            tokenizer=tokenizer
        )
        del client_model
        if args.training_algo != 'local' and 'main_train' in args.log_dir:
            path = os.path.join(args.save_dir, 'regular_clients', f'{client_id}_metrics.txt')
        elif args.training_algo == 'local':
            path = os.path.join(args.save_dir, 'local_clients', f'{client_id}_metrics.txt')
        elif (args.training_algo == 'TAP' or args.training_algo == 'DisentAFL') and 'twotimes' in args.log_dir:
            path = os.path.join(args.save_dir, 'regularAndLocal_clients', f'{client_id}_metrics.txt')
        elif (args.training_algo == 'TAP') and 'main_train' not in args.log_dir:
            path = os.path.join(args.save_dir, 'personal_clients', f'{client_id}_metrics.txt')
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
        torch.cuda.empty_cache()

        #post-training part of main-exp
        if (args.training_algo == 'TAP' or args.training_algo == 'DisentAFL') and 'main_train' in args.log_dir:
            if args.training_algo != 'DisentAFL':
                personal_model = MultiModalClientModel(
                    modalities=training_client_config['modalities'],
                    tasks=all_task_configs[group_idx],
                    embed_dim=args.embed_dim,
                    fusion=training_client_config['fusion'],
                    transformer_option=args.transformer_option,
                    personal=True,
                    domain_ids = [domain_id for _, (_, _, _, domain_id) in training_client_config['datasets'].items()],
                    task_domain_pair=[(task, modality_list, domain_id) for _, (_, task, modality_list, domain_id) in training_client_config['datasets'].items()],
                    model_type=args.model_type,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    vocab_size=tokenizer.tokenizer.vocab_size, #tokenizer.vocab_size
                ).to(args.device)
                personal_model.load_state_dict(torch.load(f'{args.log_dir}/personal_clients_postrain/personalClient{client_id}.pth'))

                metrics = evaluate_client(
                    client_model=personal_model,
                    client_data=client,
                    device=args.device,
                    batch_size=args.batch_size,
                    tokenizer=tokenizer
                )
                del personal_model
                with open(os.path.join(args.save_dir, 'personal_clients', f'{client_id}_metrics.txt'), 'w') as f:
                    json.dump(metrics, f, indent=2)
                torch.cuda.empty_cache()

            regAndLocal_model = MultiModalClientModel(
                modalities=training_client_config['modalities'],
                tasks=all_task_configs[group_idx],
                embed_dim=args.embed_dim,
                fusion=training_client_config['fusion'],
                transformer_option=args.transformer_option,
                domain_ids = [domain_id for _, (_, _, _, domain_id) in training_client_config['datasets'].items()],
                task_domain_pair=[(task, modality_list, domain_id) for _, (_, task, modality_list, domain_id) in training_client_config['datasets'].items()],
                model_type=args.model_type,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                vocab_size=tokenizer.tokenizer.vocab_size,#tokenizer.vocab_size
            ).to(args.device)
            regAndLocal_model.load_state_dict(torch.load(f'{args.log_dir}/regularAndLocal_clients/regAndLocalClient{client_id}.pth'))

            metrics = evaluate_client(
                client_model=regAndLocal_model,
                client_data=client,
                device=args.device,
                batch_size=args.batch_size,
                tokenizer=tokenizer
            )
            del regAndLocal_model
            with open(os.path.join(args.save_dir, 'regularAndLocal_clients', f'{client_id}_metrics.txt'), 'w') as f:
                json.dump(metrics, f, indent=2)
            torch.cuda.empty_cache()
        group_idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Experiment Evaluation")
    
    # Experiment setup
    parser.add_argument("--seed", type=int, default=42, help="Random seed that it was trained on")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], 
                       help="Device to use for training")
    parser.add_argument("--log_dir", type=str, default="logs", 
                       help="Directory to load checkpoints")
    parser.add_argument("--save_dir", type=str, default="logs", 
                       help="Directory to save results into a txt file.")
    
    # Federated learning parameters
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size for client evaluation")
    parser.add_argument("--num_clients", type=int, default=30, 
                       help="The number of clients to be evaluated.")
    
    # Model parameters
    parser.add_argument("--embed_dim", type=int, default=768, 
                       help="Embedding dimension for shared space")
    
    # Training Algorithm
    parser.add_argument("--training_algo", type=str, default='TAP', 
                       help="The method being evaluated")
    parser.add_argument("--transformer_option", type=str, default='gated', 
                       help="The transformer architecture to utilize")
    
    # model architecture
    parser.add_argument("--model_type", type=str, default='FLAVA',
                        help="The type of model to use.")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Number of layers to use in transformer backbone.")
    parser.add_argument("--num_heads", type=int, default=12,
                             help="The number of heads to use in each layer of backbone")

    args = parser.parse_args()
    
    # Run experiment
    run_evaluation(args)