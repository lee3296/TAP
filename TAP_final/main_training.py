import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
import json
import os
from collections import defaultdict
import copy
import time 

from models.create_model import MultiModalClientModel
from models.server_model import ServerModel 
from utils.training import train_client, train_finetune
from utils.load_clients import load_client_datasets
from utils.process_datasets import process_client_datasets
from transformers import FlavaProcessor, ViltProcessor

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
        
        for _, task in enumerate(config['tasks']):
            if task in ['cifar100_image_classification', 'imagenet_image_classification',
                        'agnews_text_classification']:
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
                    'max_length': 256 if model_type.lower() == 'flava' else 40 # ViLT has max of 40
                }
            else:
                raise ValueError(f"Invalid task of {task} has been inputted.")
        
        for dataset in config['datasets'].values():
            dataset_sizes[client_id] += len(dataset[0])
        all_task_configs[client_id] = tasks_config
        clients[client_id] = {
            'datasets': config['datasets'],
            'dataset_names': config['dataset_names'],
            'modalities': config['modalities'],
            'domain_ids': [domain_id for _, (_, _, _, domain_id) in config['datasets'].items()],
            'tasks': config['tasks'],
            'task_domain_pair': [(task, modality_list, domain_id) for _, (_, task, modality_list, domain_id) in config['datasets'].items()],
            'fusion': config['fusion']
        }
        print(f'Client {client_id} has been initalized.')
    return clients, all_task_configs, dataset_sizes

def run_federated_learning(args):
    # Set up logging
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    if args.training_algo != 'local':
        os.makedirs(os.path.join(log_dir, 'regular_clients'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'regularAndLocal_clients'), exist_ok=True)
    if args.training_algo == 'TAP':
        os.makedirs(os.path.join(log_dir, 'personal_clients'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'personal_clients_postrain'), exist_ok=True) 
    if args.training_algo == 'local':
        os.makedirs(os.path.join(log_dir, 'local_clients'), exist_ok=True)
    
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    set_seeds(args.seed)

    tokenizer = FlavaProcessor.from_pretrained("facebook/flava-full")
    if args.model_type == 'ViLT':
        tokenizer = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    
    # Load data
    client_datasets = process_client_datasets(load_client_datasets(seed=args.seed, model_type=args.model_type), tokenizer=tokenizer, model_type=args.model_type)
    
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
    
    if args.training_algo != 'local':
        server = ServerModel(
            embed_dim=args.embed_dim,
            seed=args.seed,
            tokenizer=tokenizer,
            model_type=args.model_type
        )
    set_seeds(seed=args.seed)
    
    # Register clients with server
    classification_dict = {"sh0416/ag_news": 4,
                           "uoft-cs/cifar100": 100, 'zh-plus/tiny-imagenet': 200}
    clients, all_task_configs, dataset_sizes = initialize_clients(client_datasets, classification_dict, tokenizer, args.model_type)
    if args.training_algo == 'TAP':
        personal_clients = copy.deepcopy(clients)

    for client_id, client in clients.items():
        fusion_groups = None if client_datasets[client_id]['fusion'] is False else client_datasets[client_id]['fusion-groups']
        if args.training_algo != 'local':
            server.register_client(
                client_id=client_id,
                modalities=client['modalities'],
                tasks=all_task_configs[client_id],
                domain_ids=client['domain_ids'],
                dataset_size=dataset_sizes[client_id],
                fusion_groups=fusion_groups,
                task_domain_pair=client['task_domain_pair']
            )

        if args.training_algo == 'TAP': #save personal models on disk to limit # of models on CPU/GPU
            set_seeds(args.seed)
            personal_client_config = client_datasets[client_id]

            personal_model = MultiModalClientModel(
                modalities=personal_client_config['modalities'],
                tasks=all_task_configs[client_id],
                embed_dim=args.embed_dim,
                fusion=personal_client_config['fusion'],
                transformer_option=args.transformer_option,
                personal=True,
                domain_ids = [domain_id for _, (_, _, _, domain_id) in personal_client_config['datasets'].items()],
                task_domain_pair=[(task, modality_list, domain_id) for _, (_, task, modality_list, domain_id) in personal_client_config['datasets'].items()],
                seed=args.seed,
                model_type=args.model_type,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                vocab_size=tokenizer.tokenizer.vocab_size,
            )
            checkpoint_path_personal = os.path.join(log_dir, 'personal_clients', f"personalClient{client_id}.pth")
            torch.save(personal_model.state_dict(), checkpoint_path_personal)
            del personal_model, checkpoint_path_personal
        elif args.training_algo == 'local':
            set_seeds(args.seed)
            local_config = client_datasets[client_id]
            local_model = MultiModalClientModel(
                modalities=local_config['modalities'],
                tasks=all_task_configs[client_id],
                embed_dim=args.embed_dim,
                fusion=local_config['fusion'],
                transformer_option=args.transformer_option,
                domain_ids = [domain_id for _, (_, _, _, domain_id) in local_config['datasets'].items()],
                task_domain_pair=[(task, modality_list, domain_id) for _, (_, task, modality_list, domain_id) in local_config['datasets'].items()],
                seed=args.seed,
                model_type=args.model_type,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                vocab_size=tokenizer.tokenizer.vocab_size, # tokenizer.vocab_size
            )
            checkpoint_path_local = os.path.join(log_dir, 'local_clients', f"localClient{client_id}.pth")
            torch.save(local_model.state_dict(), checkpoint_path_local)
            del local_model, checkpoint_path_local

    if args.training_algo != 'local':
        server._init_transformer(modalities=all_modalities, tasks=all_tasks, domains=all_domains, \
                                transformer_option=args.transformer_option, 
                                model_type=args.model_type, num_layers=args.num_layers, num_heads=args.num_heads)
        server.print_model_stats()
    torch.cuda.empty_cache()
    start_time = time.time()
    

    teacher_history = {
        client_id: {task: float('inf') for task in all_task_configs[client_id]}
        for client_id in range(len(clients))
    }
    student_history = {
        client_id: {task: float('inf') for task in all_task_configs[client_id]}
        for client_id in range(len(clients))
    }
    default_component = {
        'encoders': set(),
        'mote_transformers': set(),
        'mome_transformers': set(),
        'mote_experts': set(),
        'mome_experts': set(),
        'shared_proj': False
    }
    components = {
        client_id: {
            task: copy.deepcopy(default_component) 
            for task in all_task_configs[client_id]
        } for client_id in range(len(clients))
    }

    # Federated training loop
    set_seeds(seed=args.seed)
    if 'ablation_no_distill' not in args.log_dir and 'twotimes' not in args.log_dir: 
        for round_num in range(args.num_rounds):        
            # Select clients for this round
            random.seed(args.seed * round_num) 
            selected_clients = random.sample(
                list(clients.keys()),
                min(args.clients_per_round, len(clients))
            )
            print(f"\n=== Round {round_num + 1}/{args.num_rounds}, Clients: {selected_clients} ===")
                
            # Client updates
            client_updates = {}
            for client_id in tqdm(selected_clients, desc="Training clients"):
                client = clients[client_id]
                if args.training_algo == 'TAP':
                    personal_client = personal_clients[client_id]

                training_client_config = client_datasets[client_id]
                if round_num == 0 and args.training_algo != 'local':
                    set_seeds(args.seed)
                    client_model = MultiModalClientModel(
                        modalities=training_client_config['modalities'],
                        tasks=all_task_configs[client_id],
                        embed_dim=args.embed_dim,
                        fusion=training_client_config['fusion'],
                        transformer_option=args.transformer_option,
                        domain_ids = [domain_id for _, (_, _, _, domain_id) in training_client_config['datasets'].items()],
                        task_domain_pair=[(task, modality_list, domain_id) for _, (_, task, modality_list, domain_id) in training_client_config['datasets'].items()],
                        seed=args.seed,
                        model_type=args.model_type,
                        num_layers=args.num_layers,
                        num_heads=args.num_heads,
                        vocab_size=tokenizer.tokenizer.vocab_size, # tokenizer.vocab_size
                    ).to(args.device)
                elif args.training_algo != 'local':
                    client_model = server.get_global_model(client_id=client_id,
                                                        model_type=args.model_type, num_layers=args.num_layers,
                                                        num_heads=args.num_heads,
                                                        fusion=client_datasets[client_id]['fusion'])
                else: #when algo is local
                    local_config = client_datasets[client_id]
                    client_model = MultiModalClientModel(
                        modalities=local_config['modalities'],
                        tasks=all_task_configs[client_id],
                        embed_dim=args.embed_dim,
                        fusion=local_config['fusion'],
                        transformer_option=args.transformer_option,
                        domain_ids = [domain_id for _, (_, _, _, domain_id) in local_config['datasets'].items()],
                        task_domain_pair=[(task, modality_list, domain_id) for _, (_, task, modality_list, domain_id) in local_config['datasets'].items()],
                        model_type=args.model_type,
                        num_layers=args.num_layers,
                        num_heads=args.num_heads,
                        vocab_size=tokenizer.tokenizer.vocab_size,#tokenizer.vocab_size
                    )
                    client_model.load_state_dict(torch.load(os.path.join(log_dir, 'local_clients', f"localClient{client_id}.pth")))
                    client_model = client_model.to(args.device)

                # Train client, not personalized
                useDisentAFL = False
                if args.training_algo == 'DisentAFL':
                    useDisentAFL = True
                set_seeds(args.seed * round_num)
                train_client(
                    client_model=client_model,
                    client_data=client,
                    client_id=client_id,
                    device=args.device,
                    batch_size=args.batch_size,
                    accumulation_steps=args.accumulation_steps,
                    local_rounds=args.local_rounds,
                    round_num=round_num,
                    teacher_model=None,
                    personalized=False,
                    teacher_history=teacher_history[client_id],
                    components=components,
                    useDisentAFL=useDisentAFL,
                    tokenizer=tokenizer,
                    min_lr=args.min_lr,
                    peak_lr=args.peak_lr
                )
                if args.training_algo == 'local':
                    checkpoint_path_local = os.path.join(log_dir, 'local_clients', f"localClient{client_id}.pth")
                    local_cpu_model = copy.deepcopy(client_model).cpu().state_dict()
                    torch.save(local_cpu_model, checkpoint_path_local)
                    del local_cpu_model
                torch.cuda.empty_cache()
                
                #personalized client model
                if args.training_algo == 'TAP':

                    personal_client_config = client_datasets[client_id]
                    p_model = MultiModalClientModel(
                        modalities=personal_client_config['modalities'],
                        tasks=all_task_configs[client_id],
                        embed_dim=args.embed_dim,
                        fusion=personal_client_config['fusion'],
                        transformer_option=args.transformer_option,
                        personal=True,
                        domain_ids = [domain_id for _, (_, _, _, domain_id) in personal_client_config['datasets'].items()],
                        task_domain_pair=[(task, modality_list, domain_id) for _, (_, task, modality_list, domain_id) in personal_client_config['datasets'].items()],
                        model_type=args.model_type,
                        num_layers=args.num_layers,
                        num_heads=args.num_heads,
                        vocab_size=tokenizer.tokenizer.vocab_size, #tokenizer.vocab_size
                    )
                    p_model.load_state_dict(torch.load(os.path.join(log_dir, 'personal_clients', f"personalClient{client_id}.pth")))
                    p_model = p_model.to(args.device)

                    set_seeds(args.seed * round_num)
                    train_client(
                        client_model=p_model, 
                        client_data=personal_client,
                        client_id=client_id,
                        device=args.device,
                        batch_size=args.batch_size,
                        accumulation_steps=args.accumulation_steps,
                        local_rounds=args.local_rounds,
                        round_num=round_num,
                        teacher_model=client_model,
                        personalized=True,
                        margin=args.margin,
                        smaller_margin=args.smaller_margin,
                        teacher_history=teacher_history[client_id],
                        student_history=student_history[client_id],
                        components=components,
                        tokenizer=tokenizer,
                        min_lr=args.min_lr,
                        peak_lr=args.peak_lr
                    )
                    checkpoint_path_personal = os.path.join(log_dir, 'personal_clients', f"personalClient{client_id}.pth")
                    cpu_model = copy.deepcopy(p_model).cpu().state_dict() 
                    torch.save(cpu_model, checkpoint_path_personal)
                    del cpu_model, p_model
                    torch.cuda.empty_cache()
                
                # Get client update
                client_updates[client_id] = client_model.state_dict() if args.training_algo != 'local' else None
                del client_model
            
            if args.training_algo != 'local':
                for client_id, update in client_updates.items():
                    server.receive_client_update(client_id, update)
                server.aggregate_updates()
            torch.cuda.empty_cache()

        if args.training_algo != 'local': 
            for id_idx, client_id in enumerate(clients):
                checkpoint_path = os.path.join(log_dir, 'regular_clients', f"finalClient{id_idx}.pth")
                cpu_model = server.get_global_model(client_id=client_id,
                                                    model_type=args.model_type, num_layers=args.num_layers,
                                                    num_heads=args.num_heads,
                                                    fusion=client_datasets[client_id]['fusion']).cpu().state_dict()
                torch.save(cpu_model, checkpoint_path)
                del cpu_model
        torch.cuda.empty_cache()
    
    if args.training_algo == 'TAP' or args.training_algo == 'DisentAFL':
        use_DisentAFL = False if args.training_algo != 'DisentAFL' else True
        for client_id in tqdm(list(clients.keys()), desc="Afrer FL post-training....."):

            ## baseline version of post training ##
            training_client_config = client_datasets[client_id]
            teacher_model = MultiModalClientModel(
                modalities=training_client_config['modalities'],
                tasks=all_task_configs[client_id],
                embed_dim=args.embed_dim,
                fusion=training_client_config['fusion'],
                transformer_option=args.transformer_option,
                domain_ids = [domain_id for _, (_, _, _, domain_id) in training_client_config['datasets'].items()],
                task_domain_pair=[(task, modality_list, domain_id) for _, (_, task, modality_list, domain_id) in training_client_config['datasets'].items()],
                model_type=args.model_type,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                vocab_size=tokenizer.tokenizer.vocab_size,#tokenizer.vocab_size
            )
            if args.use_alt_path != 1:
                teacher_model.load_state_dict(torch.load(os.path.join(log_dir, 'regular_clients', f'finalClient{client_id}.pth')))
            elif args.use_alt_path == 1 and 'twotimes' in args.log_dir:
                teacher_model.load_state_dict(torch.load(os.path.join(args.alternative_postrain_path, 'regular_clients', f'finalClient{client_id}.pth')))
            elif args.use_alt_path == 1 and 'no_distill' in args.log_dir:
                teacher_model.load_state_dict(torch.load(os.path.join(args.alternative_postrain_path, 'personal_clients', f'personalClient{client_id}.pth')))
            teacher_model = teacher_model.to(args.device)

            set_seeds(args.seed * client_id)
            train_finetune(
                client_model=teacher_model,
                client_data=clients[client_id],
                device=args.device,
                batch_size=args.batch_size,
                accumulation_steps=args.accumulation_steps,
                distil_loss_weight=args.distil_loss_weight,
                seed=args.seed,
                total_iterations=args.distil_start_point,
                useDisentAFL=use_DisentAFL,
                tokenizer=tokenizer,
                min_lr=args.min_lr,
                peak_lr=args.peak_lr
            )

            if (args.training_algo == 'TAP') and args.use_alt_path != 1: ## proposed method of posttraining ##
                personal_model = None
                personal_client_config = client_datasets[client_id]
                personal_model = MultiModalClientModel(
                    modalities=personal_client_config['modalities'],
                    tasks=all_task_configs[client_id],
                    embed_dim=args.embed_dim,
                    fusion=personal_client_config['fusion'],
                    transformer_option=args.transformer_option,
                    personal=True,
                    domain_ids = [domain_id for _, (_, _, _, domain_id) in personal_client_config['datasets'].items()],
                    task_domain_pair=[(task, modality_list, domain_id) for _, (_, task, modality_list, domain_id) in personal_client_config['datasets'].items()],
                    model_type=args.model_type,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    vocab_size=tokenizer.tokenizer.vocab_size,#tokenizer.vocab_size
                )
                personal_model.load_state_dict(torch.load(os.path.join(log_dir, 'personal_clients', f"personalClient{client_id}.pth")))
                personal_model = personal_model.to(args.device)
                set_seeds(args.seed * client_id)
                train_finetune(
                    client_model=personal_model,
                    client_data=personal_clients[client_id],
                    device=args.device,
                    batch_size=args.batch_size,
                    accumulation_steps=args.accumulation_steps,
                    distil_loss_weight=args.distil_loss_weight,
                    teacher_model=teacher_model,
                    personalized=True,
                    seed=args.seed,
                    total_iterations=args.distil_start_point,
                    tokenizer=tokenizer,
                    min_lr=args.min_lr,
                    peak_lr=args.peak_lr
                )
                checkpoint_path_personal = os.path.join(log_dir, 'personal_clients_postrain', f"personalClient{client_id}.pth")
                cpu_model = copy.deepcopy(personal_model).cpu().state_dict()
                torch.save(cpu_model, checkpoint_path_personal)
                del cpu_model, personal_model

            ## baseline or ablation_no_distill / twotimes ###
            if args.use_alt_path != 1 or 'twotimes' in args.log_dir:
                checkpoint_path_regAndLocal = os.path.join(log_dir, 'regularAndLocal_clients', f"regAndLocalClient{client_id}.pth")
            else: #no_distill
                checkpoint_path_regAndLocal = os.path.join(log_dir, 'personal_clients_postrain', f"personalClient{client_id}.pth")
            cpu_model = copy.deepcopy(teacher_model).cpu().state_dict()
            torch.save(cpu_model, checkpoint_path_regAndLocal)
            del cpu_model, teacher_model
            torch.cuda.empty_cache()

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time
    print(f"Training executed in {elapsed_time:.4f} seconds")
    print(f"\nTraining complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Experiment")
    
    # Experiment setup
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], 
                       help="Device to use for training")
    parser.add_argument("--log_dir", type=str, default="logs", 
                       help="Directory to save logs and checkpoints")
    
    # Federated learning parameters
    parser.add_argument("--num_rounds", type=int, default=200, 
                       help="Number of federated rounds")
    parser.add_argument("--clients_per_round", type=int, default=2, 
                       help="Number of clients selected per round")
    parser.add_argument("--local_rounds", type=int, default=20, 
                       help="Number of local training iterations per client")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size for client training")
    parser.add_argument("--accumulation_steps", type=int, default=4,
                        help="The number of gradient accumulation steps")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, 
                       help="Save checkpoint every N rounds")
    
    # Model parameters
    parser.add_argument("--embed_dim", type=int, default=768, 
                       help="Embedding dimension for shared space")
    
    # Training Algorithm
    parser.add_argument("--training_algo", type=str, default='TAP', 
                       help="The method of training")
    parser.add_argument("--transformer_option", type=str, default='gated', 
                       help="The transformer architecture to utilize")
    
    # Loss weights
    parser.add_argument("--distil_loss_weight", type=float, default=1e-4, 
                       help="Weight for knowledge distillation")
    parser.add_argument("--distil_start_point", type=int, default=50, 
                       help="The number of rounds of distillation post-training")
    
    # margin weighting
    parser.add_argument("--margin", type=float, default=0.01, 
                       help="Margin for replacement")
    parser.add_argument("--smaller_margin", type=float, default=0.005, 
                       help="Margin for replacement")
    
    # model architecture
    parser.add_argument("--model_type", type=str, default='FLAVA',
                        help="The type of model to use.")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Number of layers to use in transformer backbone.")
    parser.add_argument("--num_heads", type=int, default=12,
                             help="The number of heads to use in each layer of backbone")
    
    # learning rates
    parser.add_argument("--min_lr", type=float, default=1e-4,
                        help="The minimum learning rate, start of warmup lr")
    parser.add_argument("--peak_lr", type=float, default=3e-4,
                        help="The learning rate after linear warmup")
    
    parser.add_argument("--use_alt_path", type=int, default=0,
                        help="whether to use an alternative already saved model for postrain")
    parser.add_argument("--alternative_postrain_path", type=str, default='no_path',
                        help="path of alternative models with use_alt_path is 1")
    
    args = parser.parse_args()
    # Run experiment
    run_federated_learning(args)