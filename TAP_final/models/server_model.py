import torch
import torch.nn as nn
import random
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
import copy

from .create_model import ImageEncoder, TextEncoder, \
    ImageGenerationHead, SummarizationHead, \
    GatedTransformer, MultiModalClientModel

from collections import OrderedDict

def remove_prefix(state_dict, prefix):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]  # remove prefix
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

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

class ServerModel:
    def __init__(self, embed_dim=768, seed=42, tokenizer=None, model_type='FLAVA'):
        self.embed_dim = embed_dim
        self.seed = seed
        self.clients = {} 
        self.modality_encoders = nn.ModuleDict()  # Shared encoders for all clients
        self.task_heads = nn.ModuleDict()  # Shared task heads
        self.tokenizer=tokenizer
        self.model_type = model_type
        
        # Tracking which clients use which components
        self.encoder_clients = defaultdict(set)  # {encoder_name: set(client_ids)}
        self.fusion_clients = defaultdict(set)   # {client_id: set(fusion_key(s))}
        self.task_clients = defaultdict(set)     # {task_name: set(client_ids)}
        
        # State dictionaries for aggregation
        self.encoder_states = {}
        self.transformer_state = None#{}
        self.task_states = {}
        
        # Counters for weighted averaging
        self.encoder_counts = defaultdict(int)
        self.task_counts = defaultdict(int)

    def register_client(self, client_id: int, 
                       modalities: List[str], 
                       tasks: Dict[str, dict],
                       domain_ids: List[str],
                       dataset_size: int,
                       fusion_groups=None,
                       task_domain_pair=None):
        """Register a new client with its modalities and tasks"""
        if client_id in self.clients:
            raise ValueError(f"Client {client_id} already registered")
            
        self.clients[client_id] = {
            'modalities': modalities,
            'tasks': tasks,
            'dataset_size': dataset_size,
            'domain_ids': domain_ids,
            'task_domain_pair': task_domain_pair,
        }
        
        # Register encoders needed by this client
        for modality in modalities:
            if modality not in self.modality_encoders:
                self._init_encoder(modality)
            self.encoder_clients[modality].add(client_id)
        
        # Register cross-modal client that fuse.
        if fusion_groups:
            for fusion_group in fusion_groups:
                modality_1, modality_2 = fusion_group.split("-")
                fusion_key = f"{modality_1}-{modality_2}"
                self.fusion_clients[client_id].add(fusion_key)
        
        # Register task heads
        for task_name, task_params in tasks.items(): 
            if task_name not in self.task_heads:
                self._init_task_head(task_name, task_params)
            self.task_clients[task_name].add(client_id)
    
    def _init_transformer(self, modalities:list, tasks:list, domains:list, transformer_option: str, 
                          model_type: str, num_layers: int, num_heads: int):
        self.transformer_option = transformer_option
        self.fusion = GatedTransformer(embed_dim=self.embed_dim, modalities=modalities, 
                                    tasks=tasks, domain_ids=domains, seed=self.seed, 
                                    model_type=model_type, num_layers=num_layers, num_heads=num_heads)
        self.transformer_counts = {}

    def _init_encoder(self, modality: str):
        """Initialize the appropriate encoder for a modality"""
        if modality == 'image':
            encoder = ImageEncoder(embed_dim=self.embed_dim, seed=self.seed, model_type=self.model_type)
        elif modality == 'text':
            encoder = TextEncoder(embed_dim=self.embed_dim, use_lora=False, model_type=self.model_type, seed=self.seed, vocab_size=self.tokenizer.tokenizer.vocab_size)#self.tokenizer.vocab_size)
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        self.modality_encoders[modality] = encoder
        self.encoder_states[modality] = None
        self.encoder_counts[modality] = 0

    def _init_task_head(self, task_name: str, task_params: dict):
        """Initialize a task-specific head"""
        if task_name in ['agnews_text_classification', 'cifar100_image_classification',
                        'imagenet_image_classification']:
            set_seeds(seed=self.seed)
            if self.model_type == 'FLAVA':
                head = nn.Sequential(
                        nn.Linear(self.embed_dim, self.embed_dim // 2), 
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(self.embed_dim // 2, task_params['num_classes']), 
                )
            else:
                head = nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim * 4), 
                    nn.LayerNorm(self.embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(0.1), 
                    nn.Linear(self.embed_dim * 4, self.embed_dim),
                    nn.GELU(),
                    nn.Linear(self.embed_dim, task_params['num_classes']),
                )
            nn.init.xavier_uniform_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)  
        elif task_params['type'] == 'generation':
            if task_params.get('modality') == 'image':
                head = ImageGenerationHead(
                    embed_dim=self.embed_dim,
                    img_size=task_params.get('img_size', 32),
                    channels=task_params.get('channels', 3),
                    seed=self.seed
                )
            else:
                head = SummarizationHead(
                    embed_dim=self.embed_dim,
                    vocab_size=task_params['vocab_size'],
                    max_length=task_params.get('max_length', 256),
                    use_lora=False,
                    seed=self.seed
                )
        else:
            raise ValueError(f"Unknown task type: {task_name}")
        
        self.task_heads[task_name] = head
        self.task_states[task_name] = None
        self.task_counts[task_name] = 0

    def receive_client_update(self, client_id: int, 
                            client_model_state: dict):
        """
        Receive an update from a client and prepare for aggregation
        
        Args:
            client_id: ID of the client sending the update
            client_model_state: state_dict of the client's model
        """
        client_info = self.clients[client_id]
        
        # Process encoders
        for modality in client_info['modalities']:
            encoder_state = {k: v for k, v in client_model_state.items() 
                           if k.startswith(f'encoders.{modality}')}
            
            if self.encoder_states[modality] is None:
                self.encoder_states[modality] = {k: v.clone() * self.clients[client_id]['dataset_size']
                                                for k, v in encoder_state.items()}
            else:
                for k in self.encoder_states[modality]:
                    self.encoder_states[modality][k] += encoder_state[k] * self.clients[client_id]['dataset_size'] 
            self.encoder_counts[modality] += self.clients[client_id]['dataset_size'] 
        

        # Process transformer
        if self.transformer_option == 'gated':
            transformer_state = {
                k: v for k, v in client_model_state.items()
                if (k.startswith('fusion.mote_experts.') or 
                    k.startswith('fusion.mome_experts.') or
                    k.startswith('fusion.mote_transformers.') or
                    k.startswith('fusion.mome_transformers.') or
                    k.startswith('fusion.shared_proj'))
            }
            if self.transformer_state is None:
                self.transformer_state = {
                    k: v.clone() * self.clients[client_id]['dataset_size']
                    for k, v in transformer_state.items()
                }
                self.transformer_counts = {
                    k: self.clients[client_id]['dataset_size']
                    for k in transformer_state
                }
            else:
                for k, v in transformer_state.items():
                    if k in self.transformer_state:
                        self.transformer_state[k] += v * self.clients[client_id]['dataset_size']
                        self.transformer_counts[k] += self.clients[client_id]['dataset_size']
                    else:
                        # New parameter that wasn't seen before
                        self.transformer_state[k] = v * self.clients[client_id]['dataset_size']
                        self.transformer_counts[k] = self.clients[client_id]['dataset_size']
        else: 
            raise ValueError("Provided transformer architecture is not valid!")
        
        # Process task heads
        for task_name in client_info['tasks']:
            task_state = {k: v for k, v in client_model_state.items() 
                        if k.startswith(f'head.task_heads.{task_name}')}
            
            if self.task_states[task_name] is None:
                self.task_states[task_name] = {k: v.clone() * self.clients[client_id]['dataset_size'] # num_samples 
                                             for k, v in task_state.items()}
            else:
                for k in self.task_states[task_name]:
                    self.task_states[task_name][k] += task_state[k] * self.clients[client_id]['dataset_size'] # num_samples
            self.task_counts[task_name] += self.clients[client_id]['dataset_size'] # num_samples

    def aggregate_updates(self):
        """Aggregate all received updates using weighted averaging"""
        # Aggregate encoders
        for modality in self.modality_encoders:
            if self.encoder_counts[modality] > 0:
                for k in self.encoder_states[modality]:
                    self.encoder_states[modality][k] = self.encoder_states[modality][k].float() / self.encoder_counts[modality] 
                self.encoder_states[modality] = remove_prefix(self.encoder_states[modality], f"encoders.{modality}.")
                self.modality_encoders[modality].load_state_dict(self.encoder_states[modality])
                self.encoder_states[modality] = None
                self.encoder_counts[modality] = 0
        
        # Aggregate transformer
        if self.transformer_option == 'gated':
            if self.transformer_counts:  # checks if dict is not empty
                for k in self.transformer_state:
                    count = self.transformer_counts.get(k, 0)
                    if count > 0:
                        self.transformer_state[k] /= count
                    else:
                        print(f"Warning: No count found for parameter {k}, skipping normalization.")
                self.transformer_state = remove_prefix(self.transformer_state, "fusion.")
                self.fusion.load_state_dict(self.transformer_state, strict=False)
                # Reset for next round
                self.transformer_state = None
                self.transformer_counts = {}
        else:
            raise ValueError("Provided transformer architecture is not valid!")
        
        # Aggregate task heads
        for task_name in self.task_heads:
            if self.task_counts[task_name] > 0:
                for k in self.task_states[task_name]:
                    self.task_states[task_name][k] = self.task_states[task_name][k].float() / self.task_counts[task_name] 
                state_dict = remove_prefix(self.task_states[task_name], f"head.task_heads.{task_name}.")
                self.task_heads[task_name].load_state_dict(state_dict)
                self.task_states[task_name] = None
                self.task_counts[task_name] = 0

    def get_global_model(self, client_id: Optional[int] = None,
                         model_type: Optional[str] = 'FLAVA', num_layers: Optional[int] = 2, 
                         num_heads: Optional[int] = 4, fusion: Optional[bool] = False):
        """
        Get a complete model for a specific client.
        """
        if client_id is None:
            # Return complete server model (for inspection)
            return copy.deepcopy({
                'modality_encoders': self.modality_encoders,
                'transformer': self.fusion,
                'task_heads': self.task_heads
            })
        else:
            # Create a client-specific model
            if client_id not in self.clients:
                raise ValueError(f"Client {client_id} not registered")
            
            client_info = self.clients[client_id]
            model = MultiModalClientModel(
                modalities=client_info['modalities'],
                tasks=client_info['tasks'],
                embed_dim=self.embed_dim,
                transformer_option=self.transformer_option,
                domain_ids=client_info['domain_ids'],
                task_domain_pair=client_info['task_domain_pair'],
                fusion=fusion,
                model_type=model_type,
                num_layers=num_layers,
                num_heads=num_heads,
                vocab_size=self.tokenizer.tokenizer.vocab_size,#self.tokenizer.vocab_size
            )
            
            # Load the shared components
            state_dict = {}
            
            # Load encoders
            for modality in client_info['modalities']:
                state_dict.update({
                    f'encoders.{modality}.{k}': v 
                    for k, v in self.modality_encoders[modality].state_dict().items()
                })
            
            # update transformer
            if self.transformer_option == 'gated':
                submodules = self.fusion.get_submodules(
                    modalities=self.clients[client_id]['modalities'],
                    tasks=self.clients[client_id]['tasks'],
                )
                # Handle MoTE experts
                for layer_key, layer_dict in submodules['mote_experts'].items():
                    for mod, task_experts in layer_dict.items():
                        for task, expert in task_experts.items():
                            for k, v in expert.state_dict().items():
                                state_dict[f'mote_experts.{layer_key}.{mod}.{task}.{k}'] = v
                # Handle MoME experts
                for layer_key, layer_dict in submodules['mome_experts'].items():
                    for task, mod_experts in layer_dict.items():
                        for mod, expert in mod_experts.items():
                            for k, v in expert.state_dict().items():
                                state_dict[f'mome_experts.{layer_key}.{task}.{mod}.{k}'] = v
                # Handle transformer blocks
                for mod, transformer in submodules['mote_transformers'].items():
                    for k, v in transformer.state_dict().items():
                        state_dict[f'mote_transformers.{mod}.{k}'] = v
                for task, transformer in submodules['mome_transformers'].items():
                    for k, v in transformer.state_dict().items():
                        state_dict[f'mome_transformers.{task}.{k}'] = v
                # Handle projection layers
                for k, v in submodules['shared_proj'].state_dict().items():
                    state_dict[f'shared_proj.{k}'] = v
            else:
                raise ValueError("Provided transformer architecture is not valid!")
            
            # Load task heads
            for task_name in client_info['tasks']:
                state_dict.update({
                    f'head.task_heads.{task_name}.{k}': v 
                    for k, v in self.task_heads[task_name].state_dict().items()
                })
            
            model.load_state_dict(state_dict, strict=False)
            return model

    def print_model_stats(self):
        """Print statistics about the server model"""
        print("=== Server Model Statistics ===")
        print(f"Registered clients: {len(self.clients)}")
        
        total_params = sum(p.numel() for p in self.modality_encoders.parameters())
        total_params += sum(p.numel() for p in self.fusion.parameters())
        total_params += sum(p.numel() for p in self.task_heads.parameters())
        
        trainable_params = sum(p.numel() for p in self.modality_encoders.parameters() if p.requires_grad)
        trainable_params += sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
        trainable_params += sum(p.numel() for p in self.task_heads.parameters() if p.requires_grad)
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params/total_params:.2f}%")