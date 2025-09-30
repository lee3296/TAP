"""
Enhanced multi-modal, multi-task framework
"""
import math
import random
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from peft import LoraConfig, inject_adapter_in_model, get_peft_model



from transformers import FlavaModel, ViltModel
# ======================== Helper Functions =============================
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


def make_expert_with_ffn(embed_dim, model_type="FLAVA", layer_idx=0, seed=42):
    """
    Creates an expert MLP and loads weights from FLAVA or ViLT transformer layer.

    Args:
        embed_dim (int): Model hidden size.
        model_type (str): Either "flava" or "vilt".
        layer_idx (int): Encoder layer index to copy weights from.
        seed (int): Random seed.

    Returns:
        nn.Module: LoRA-augmented expert MLP with frozen base weights.
    """
    # Create base feed-forward network
    expert = nn.Sequential(
        nn.Linear(embed_dim, embed_dim * 4),
        nn.GELU(),
        nn.Linear(embed_dim * 4, embed_dim)
    )

    # Load source model and state_dict based on model_type
    rank = 4 #flava
    lora_alpha = 8 #flava
    if model_type.lower() == "flava":
        model = FlavaModel.from_pretrained("facebook/flava-full")
        prefix = f"multimodal_model.encoder.layer.{layer_idx}."
        intermediate_w = f"{prefix}intermediate.dense.weight"
        intermediate_b = f"{prefix}intermediate.dense.bias"
        output_w = f"{prefix}output.dense.weight"
        output_b = f"{prefix}output.dense.bias"
    elif model_type.lower() == "vilt":
        rank = 8
        lora_alpha = 16
        model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        prefix = f"encoder.layer.{layer_idx}."
        intermediate_w = f"{prefix}intermediate.dense.weight"
        intermediate_b = f"{prefix}intermediate.dense.bias"
        output_w = f"{prefix}output.dense.weight"
        output_b = f"{prefix}output.dense.bias"
    else:
        raise ValueError(f"Unknown model_type '{model_type}', must be 'flava' or 'vilt'.")

    state_dict = model.state_dict()

    # Copy weights into FFN
    with torch.no_grad():
        expert[0].weight.copy_(state_dict[intermediate_w])
        expert[0].bias.copy_(state_dict[intermediate_b])
        expert[2].weight.copy_(state_dict[output_w])
        expert[2].bias.copy_(state_dict[output_b])

    # Cleanup
    del model, state_dict

    # LoRA config and injection
    config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=["0", "2"],  # Linear layers in Sequential
        lora_dropout=0.3,
        bias="none"
    )
    expert = inject_adapter_in_model(config, expert)
    # Freeze non-LoRA parameters
    for name, param in expert.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return expert

# ======================== Modality Encoders ============================
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=768, image_size=64, model_type='FLAVA', seed=42):
        super().__init__()
        set_seeds(seed)
        self.embed_dim = embed_dim
        self.model_type = model_type

        if model_type == "FLAVA":
            base_model = FlavaModel.from_pretrained("facebook/flava-full")
            self.backbone = base_model.image_model
            hidden_size = base_model.config.hidden_size
        elif model_type == "ViLT":
            base_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
            self.cls_token = base_model.embeddings.cls_token
            self.backbone = base_model.embeddings.patch_embeddings
            self.position_embeddings = base_model.embeddings.position_embeddings
            hidden_size = base_model.config.hidden_size
        else:
            raise ValueError(f"Unknown model_type '{model_type}', must be 'flava' or 'vilt'.")
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        if model_type == "FLAVA":
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["attention.attention.query", "attention.attention.value"],
                lora_dropout=0.3,
                bias="none"
            )
            self.backbone = get_peft_model(self.backbone, config)
        elif model_type == 'ViLT':
            config = LoraConfig(
                r=4, 
                lora_alpha=8, 
                target_modules=["projection"],  # Target the patch embedding projection
                lora_dropout=0.3,
                bias="none"
            )
            # Apply LoRA only to the projection layer
            self.backbone = get_peft_model(self.backbone, config)
        
        # Get the image to match expected dimensions
        self.input_adaptor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True),
            nn.Conv2d(3, 3, kernel_size=1)
        )
        
        # Projection to match embed_dim
        self.proj = nn.Linear(hidden_size, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.input_adaptor(x)
        if self.model_type == "FLAVA":
            outputs = self.backbone(pixel_values=x)
            embeddings = outputs.last_hidden_state
        elif self.model_type == "ViLT":
            embeddings = self.backbone(x)  # [B, hidden, H_p, W_p]
            B, C, H, W = embeddings.shape
            embeddings = embeddings.flatten(2)        # [B, C, H*W]
            embeddings = embeddings.transpose(1, 2)   # [B, H*W, C]
            cls_tokens = self.cls_token.expand(B, -1, -1)      # [B, 1, C]
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)  # prepend CLS
            embeddings = embeddings + self.position_embeddings[:, :embeddings.size(1), :]
        else:
            raise ValueError(f"Unknown model_type.")
        embeddings = self.proj(embeddings)
        return self.layer_norm(embeddings)

class TextEncoder(nn.Module):
    def __init__(self, embed_dim=768, use_lora=False, model_type='FLAVA', seed=42, vocab_size=30522):
        super().__init__()
        set_seeds(seed)
        self.embed_dim = embed_dim
        self.model_type = model_type

        if self.model_type == "FLAVA":
            base_model = FlavaModel.from_pretrained("facebook/flava-full")
            self.backbone = base_model.text_model
            hidden_size = base_model.config.hidden_size
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif self.model_type == "ViLT":
            base_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
            self.embeddings = base_model.embeddings
            hidden_size = base_model.config.hidden_size
            # Freeze all embedding parameters
            for param in self.embeddings.parameters():
                param.requires_grad = False 
            # Store individual components for easier access
            self.word_embeddings = self.embeddings.text_embeddings.word_embeddings
            self.position_embeddings = self.embeddings.text_embeddings.position_embeddings
            self.token_type_embeddings = self.embeddings.text_embeddings.token_type_embeddings
            self.text_layer_norm = self.embeddings.text_embeddings.LayerNorm
            self.dropout = self.embeddings.dropout
        else:
            raise ValueError(f"Unknown model_type '{model_type}', must be 'flava' or 'vilt'.")

        if self.model_type == 'FLAVA': # ViLT does not need LoRA since it's just embeddings
            config = LoraConfig(
                r=8, 
                lora_alpha=16,
                target_modules=["attention.attention.query", "attention.attention.value"],
                lora_dropout=0.3,
                bias="none"
            )
            self.backbone = get_peft_model(self.backbone, config)
        
        self.proj = nn.Linear(hidden_size, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids, attention_mask):
        if self.model_type == "FLAVA":
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
        elif self.model_type == "ViLT":
            # Get word embeddings
            inputs_embeds = self.word_embeddings(input_ids)
            # Create position IDs (0 to seq_length-1)
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=input_ids.device)
            embeddings = inputs_embeds
            embeddings = embeddings + self.position_embeddings(position_ids)
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)
            # Apply layer normalization and dropout
            embeddings = self.text_layer_norm(embeddings)
            embeddings = self.dropout(embeddings)
        else:
            raise ValueError(f"Unknown model_type '{self.model_type}'.")
        embeddings = self.proj(embeddings)
        return self.layer_norm(embeddings)

# ======== Transformer backbone ======== #
class MoDELayer(nn.Module):
    def __init__(self, embed_dim, domain_ids_list, layer_idx, mod, mod_or_task_name, 
                 task_domain_pair, init_from_model_type, seed=42):
        super().__init__()
        set_seeds(seed)

        domain_ids = list(dict.fromkeys(domain_ids_list))
        self.domain_ids = domain_ids
        self.experts = nn.ModuleDict()

        if init_from_model_type == 'FLAVA' or init_from_model_type == 'ViLT':
            if mod_or_task_name == 'shared' or task_domain_pair is None:
                for domain_id in domain_ids:
                    expert = make_expert_with_ffn(
                        embed_dim, layer_idx=layer_idx, 
                        seed=seed, model_type=init_from_model_type
                    )
                    self.experts[domain_id] = expert
            else:
                for (task, modality_list, domain_id) in task_domain_pair:
                    if mod and mod_or_task_name in modality_list:
                        expert = make_expert_with_ffn(
                            embed_dim, layer_idx=layer_idx, 
                            seed=seed, model_type=init_from_model_type
                        )
                        self.experts[domain_id] = expert
                    elif mod is False and (mod_or_task_name == task or mod_or_task_name in task):
                        expert = make_expert_with_ffn(
                            embed_dim, layer_idx=layer_idx, 
                            seed=seed, model_type=init_from_model_type
                        )
                        self.experts[domain_id] = expert
        else:
            # Fallback to randomly initialized experts
            for domain_id in domain_ids:
                expert = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                )
                self.experts[domain_id] = expert

    def forward(self, x, domain_id):
        expert_out = self.experts[domain_id](x)
        return x + expert_out

class LoRALayer(nn.Module):
    def __init__(self, base_linear, r=16, lora_alpha=32, lora_dropout=0.3, seed=42):
        super().__init__()
        set_seeds(seed=seed)

        self.base_linear = base_linear
        self.base_linear.requires_grad_(False)  # freeze base weights
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.lora_A = nn.Linear(base_linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_linear.out_features, bias=False)
        self.scaling = lora_alpha / r

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        orig_out = self.base_linear(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return orig_out + lora_out

class TransformerMoEBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=12, mod=True, mod_or_task_name=None, 
                 domain_ids=None, num_layers=1, task_domain_pair=None, is_encoder=True, seed=42,
                 model_type=None):
        super().__init__()
        set_seeds(seed)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.domain_ids = domain_ids
        self.is_encoder = is_encoder
        self.model_type = model_type

        # --- Load source model ---
        rank = 16 #flava
        lora_alpha = 32 #flava
        if self.model_type == "FLAVA":
            model = FlavaModel.from_pretrained("facebook/flava-full")
            encoder_layers = model.multimodal_model.encoder.layer
        elif self.model_type == "ViLT":
            rank=8
            lora_alpha=16
            model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
            encoder_layers = model.encoder.layer
        else:
            raise ValueError(f"Unknown model_type '{model_type}', must be 'flava' or 'vilt'.")
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Initialize attention
            attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
            
            # Load weights from FLAVA's multimodal encoder
            curr_i = i if is_encoder else i + num_layers
            enc_layer = encoder_layers[i] if is_encoder else encoder_layers[i + num_layers]
            
            with torch.no_grad():
                # Load attention weights
                q_weight = enc_layer.attention.attention.query.weight
                k_weight = enc_layer.attention.attention.key.weight
                v_weight = enc_layer.attention.attention.value.weight
                
                attn.in_proj_weight.data.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
                attn.in_proj_bias.data.copy_(torch.cat([
                    enc_layer.attention.attention.query.bias,
                    enc_layer.attention.attention.key.bias,
                    enc_layer.attention.attention.value.bias
                ], dim=0))
                
                attn.out_proj.weight.copy_(enc_layer.attention.output.dense.weight)
                attn.out_proj.bias.copy_(enc_layer.attention.output.dense.bias)
                
                # Initialize LoRA layers
                base_q = nn.Linear(embed_dim, embed_dim, bias=False)
                base_v = nn.Linear(embed_dim, embed_dim, bias=False)
                base_q.weight.copy_(q_weight)
                base_v.weight.copy_(v_weight)

            lora_q = LoRALayer(base_q, seed=seed, r=rank, lora_alpha=lora_alpha)
            lora_v = LoRALayer(base_v, seed=seed, r=rank, lora_alpha=lora_alpha)

            def make_new_forward(attn_layer, lora_q_layer, lora_v_layer):
                original_forward = attn_layer.forward
                def new_forward(query, key, value, *args, **kwargs):
                    query = lora_q_layer(query)
                    value = lora_v_layer(value)
                    return original_forward(query, key, value, *args, **kwargs)
                return new_forward
            attn.forward = make_new_forward(attn, lora_q, lora_v)

            # Freeze base attention
            for param in attn.parameters():
                param.requires_grad = False
            for param in lora_q.parameters():
                param.requires_grad = True
            for param in lora_v.parameters():
                param.requires_grad = True

            attn.lora_q = lora_q
            attn.lora_v = lora_v

            # Layer norms
            attention_norm = nn.LayerNorm(embed_dim)
            attention_norm.weight.data.copy_(enc_layer.layernorm_before.weight)
            attention_norm.bias.data.copy_(enc_layer.layernorm_before.bias)
            attention_norm.requires_grad_(False)

            expert_norm = nn.LayerNorm(embed_dim)
            expert_norm.weight.data.copy_(enc_layer.layernorm_after.weight)
            expert_norm.bias.data.copy_(enc_layer.layernorm_after.bias)
            expert_norm.requires_grad_(False)

            layer = nn.ModuleDict({
                'attention': attn,
                'attention_norm': attention_norm,
                'expert_norm': expert_norm,
                'mode_q': MoDELayer(embed_dim, domain_ids, layer_idx=curr_i, mod=mod, 
                                   mod_or_task_name=mod_or_task_name, 
                                   task_domain_pair=task_domain_pair, 
                                   init_from_model_type=model_type, seed=seed),
                'mode_v': MoDELayer(embed_dim, domain_ids, layer_idx=curr_i, mod=mod, 
                                   mod_or_task_name=mod_or_task_name, 
                                   task_domain_pair=task_domain_pair, 
                                   init_from_model_type=model_type, seed=seed)
            })
            self.layers.append(layer)
        
        del model, encoder_layers

    def forward(self, x, expert_models=None, domain_id=None):
        for i, layer in enumerate(self.layers):
            x_norm = layer['attention_norm'](x)
            q = k = v = x_norm
            
            if layer['mode_q']:
                q = layer['mode_q'](q, domain_id)
            if layer['mode_v']:
                v = layer['mode_v'](v, domain_id)
                
            attn_out, _ = layer['attention'](q, k, v)
            x = x + attn_out
            
            if expert_models is not None:
                x_norm = layer['expert_norm'](x)
                expert_out = expert_models[i](x_norm)
                x = x + expert_out
        return x

class GatedTransformer(nn.Module):
    def __init__(self, modalities, tasks, domain_ids, embed_dim=768, 
                 num_heads=12, num_layers=1, task_domain_pair=None,
                 seed=42, model_type='FLAVA'): 
        super().__init__()
        set_seeds(seed=seed)

        self.modalities = modalities  # List of M modalities
        self.tasks = tasks          # List of O tasks
        self.domain_ids = domain_ids # List of D domain identifiers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        task_list = tasks if isinstance(tasks, (list, tuple)) else list(tasks.keys())
        cleaned_task_list = []
        for task in task_list:
            if "text_classification" in task:
                if "text_classification" not in cleaned_task_list:
                    cleaned_task_list.append("text_classification")
            elif "image_classification" in task:
                if "image_classification" not in cleaned_task_list:
                    cleaned_task_list.append("image_classification")
            elif task not in cleaned_task_list:
                cleaned_task_list.append(task)
        task_list = cleaned_task_list

        # MoTE (Modality-Task Experts)
        self.mote_experts = nn.ModuleDict({
            f'layer_{i}': nn.ModuleDict({
                mod: nn.ModuleDict({
                    task: make_expert_with_ffn(embed_dim, layer_idx=i, 
                                                   seed=seed, model_type=model_type)
                    for task in task_list + ['shared']
                }) for mod in modalities + ['shared']
            }) for i in range(num_layers)
        })
        torch.cuda.empty_cache()
        # MoME (Modality-Modality Experts)
        self.mome_experts = nn.ModuleDict({
            f'layer_{i}': nn.ModuleDict({
                task: nn.ModuleDict({
                    mod: make_expert_with_ffn(embed_dim, layer_idx=i+num_layers, 
                                                  seed=seed, model_type=model_type)
                    for mod in modalities + ['shared']
                }) for task in task_list + ['shared']
            }) for i in range(num_layers)
        })
        torch.cuda.empty_cache()
        # Transformer blocks with built-in MoDE layers
        self.mote_transformers = nn.ModuleDict({
            mod: TransformerMoEBlock(embed_dim, num_heads, True, mod, domain_ids, num_layers=self.num_layers, 
                                     task_domain_pair=task_domain_pair, is_encoder=True, seed=seed, model_type=model_type)
            for mod in modalities + ['shared']
        })
        self.mome_transformers = nn.ModuleDict({
            task: TransformerMoEBlock(embed_dim, num_heads, False, task, domain_ids, num_layers=num_layers, 
                                      task_domain_pair=task_domain_pair, is_encoder=False, seed=seed, model_type=model_type)
            for task in task_list + ['shared']
        })
        # Projection layers
        self.shared_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs: dict, task: str, domain_id: str, useDisentAFL: bool):
        use_cls_token = False
        if 'image_classification' in task:
            task = 'image_classification'
        elif 'text_classification' in task:
            task = 'text_classification'

        all_outputs = []
        if useDisentAFL: #disentAFL, get latent for orthogonal loss
            all_disentAFL_latents = []
            latent_info = []  # To store (modality, task) pairs for each latent

        for mod, x in inputs.items():
            if mod not in self.modalities:
                raise ValueError(f"Unknown modality '{mod}'")        

            # Split into shared and private parts
            h_shared = self.shared_proj(x)  # [B, S, D]
            h_private = x  # [B, S, D]
            # 1. Shared-Shared Path
            if 'shared' in self.mote_experts:
                # MoTE processing
                ss_mote_output = self.mote_transformers['shared'](
                    h_shared, 
                    expert_models=[self.mote_experts[f'layer_{i}']['shared']['shared'] for i in range(self.num_layers)], 
                    domain_id=domain_id
                )
                # MoME processing
                ss_output = self.mome_transformers['shared'](
                    ss_mote_output,
                    expert_models=[self.mome_experts[f'layer_{i}']['shared']['shared'] for i in range(self.num_layers)],
                    domain_id=domain_id
                )
                all_outputs.append(ss_output[:, 0] if use_cls_token else ss_output.mean(dim=1))
            
            # 2. Shared-Task Path
            # MoTE processing
            st_mote_output = self.mote_transformers[mod](
                h_shared,
                expert_models=[self.mote_experts[f'layer_{i}']['shared'][task] for i in range(self.num_layers)],
                domain_id=domain_id
            )

            # MoME processing
            st_output = self.mome_transformers[task](
                st_mote_output,
                expert_models=[self.mome_experts[f'layer_{i}'][task]['shared'] for i in range(self.num_layers)],
                domain_id=domain_id
            )
            all_outputs.append(st_output[:, 0] if use_cls_token else st_output.mean(dim=1)) #[B,D]
            
            # 3. Mod-Task Path (Private)
            # MoTE processing
            mt_mote_output = self.mote_transformers[mod](
                h_private,
                expert_models=[self.mote_experts[f'layer_{i}'][mod][task] for i in range(self.num_layers)],
                domain_id=domain_id
            )

            # disentAFL baseline
            if useDisentAFL:
                all_disentAFL_latents.append(mt_mote_output)
                latent_info.append((mod, task))  # Specific modality, specific task

            # MoME processing
            mt_output = self.mome_transformers[task](
                mt_mote_output,
                expert_models=[self.mome_experts[f'layer_{i}'][task][mod] for i in range(self.num_layers)],
                domain_id=domain_id
            )
            all_outputs.append(mt_output[:, 0] if use_cls_token else mt_output.mean(dim=1))
            
            # 4. Mod-Shared Path (Private)
            # MoTE processing
            ms_mote_output = self.mote_transformers[mod](
                h_private,
                expert_models=[self.mote_experts[f'layer_{i}'][mod]['shared'] for i in range(self.num_layers)],
                domain_id=domain_id
            )
            # MoME processing
            ms_output = self.mome_transformers['shared'](
                ms_mote_output,
                expert_models=[self.mome_experts[f'layer_{i}']['shared'][mod] for i in range(self.num_layers)],
                domain_id=domain_id
            )
            all_outputs.append(ms_output[:, 0] if use_cls_token else ms_output.mean(dim=1))
        # Combine all outputs
        combined = torch.stack(all_outputs, dim=0).mean(dim=0)  # [B, D]

        if useDisentAFL:
            return combined, (all_disentAFL_latents, latent_info)
        return combined 

    def get_submodules(self, modalities=None, tasks=None):
        """Extract relevant branches, organized per-layer."""
        modalities = modalities or self.modalities
        tasks = tasks or self.tasks
        if isinstance(tasks, dict):
            tasks = list(tasks.keys())
        
        return {
            'mote_experts': {
                f'layer_{i}': {
                    m: {t: self.mote_experts[f'layer_{i}'][m][t]
                        for t in tasks + ['shared']
                        if t in self.mote_experts[f'layer_{i}'][m]}
                    for m in modalities + ['shared']
                    if m in self.mote_experts[f'layer_{i}']
                } for i in range(self.num_layers)
            },
            'mome_experts': {
                f'layer_{i}': {
                    t: {m: self.mome_experts[f'layer_{i}'][t][m]
                        for m in modalities + ['shared']
                        if m in self.mome_experts[f'layer_{i}'][t]}
                    for t in tasks + ['shared']
                    if t in self.mome_experts[f'layer_{i}']
                } for i in range(self.num_layers)
            },
            'mote_transformers': {
                m: self.mote_transformers[m]
                for m in modalities + ['shared']
                if m in self.mote_transformers
            },
            'mome_transformers': {
                t: self.mome_transformers[t]
                for t in tasks + ['shared']
                if t in self.mome_transformers
            },
            'shared_proj': self.shared_proj,
        }
    
# ======================== Task Heads ============================
class SummarizationHead(nn.Module):
    def __init__(self, embed_dim=768, vocab_size=30522, max_length=256, use_lora=False, seed=42): 
        super().__init__()
        self.max_length = max_length
        set_seeds(seed=seed)
        
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=embed_dim,
            n_layer=1,
            n_head=4, 
            n_positions=max_length,
            n_inner=embed_dim*2,
        )
        self.gpt2 = GPT2Model(config)
        
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.position_embeddings = nn.Embedding(max_length, embed_dim)
        self.cond_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

    def forward(self, x, input_features=None):
        batch_size = x.size(0)
        position_ids = torch.arange(0, self.max_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embeddings(position_ids)
        
        if input_features is not None:
            attn_output, _ = self.cond_attention(
                query=pos_embeds,
                key=input_features.unsqueeze(1),
                value=input_features.unsqueeze(1)
            )
            pos_embeds = pos_embeds + attn_output
        
        transformer_outputs = self.gpt2(
            inputs_embeds=pos_embeds,
            attention_mask=torch.ones_like(position_ids)
        )
        return self.lm_head(transformer_outputs.last_hidden_state)

class ImageGenerationHead(nn.Module):
    def __init__(self, embed_dim=768, img_size=32, channels=3, seed=42):
        super().__init__()
        self.channels = channels
        set_seeds(seed=seed)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, 64 * 8 * 8), #64
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  #16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), #32x32
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, kernel_size=4, stride=2, padding=1) #64x64
        )
    
    def forward(self, x):
        x = self.proj(x)
        x = x.view(-1, 64, 8, 8) #64,8,8
        return self.decoder(x)

# ======================== Enhanced Multi-Task Head ============================

class MultiTaskHead(nn.Module):
    def __init__(self, embed_dim=768, tasks=None, seed=42, model_type='FLAVA'):
        super().__init__()
        self.task_heads = nn.ModuleDict()
        
        # Task-specific heads
        for task, params in tasks.items():
            if task in ['agnews_text_classification', 'cifar100_image_classification',
                        'imagenet_image_classification']:
                set_seeds(seed=seed)
                if model_type == 'FLAVA':
                    head = nn.Sequential(
                        nn.Linear(embed_dim, embed_dim // 2), 
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(embed_dim // 2, params['num_classes']), 
                    )
                else:
                    head = nn.Sequential(
                        nn.Linear(embed_dim, embed_dim * 4), 
                        nn.LayerNorm(embed_dim * 4),
                        nn.GELU(),
                        nn.Dropout(0.1), 
                        nn.Linear(embed_dim * 4, embed_dim),
                        nn.GELU(),
                        nn.Linear(embed_dim, params['num_classes']),
                    )
                nn.init.xavier_uniform_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)  
                self.task_heads[task] = head
            elif params['type'] == 'generation': # all other tasks can be done by 1 of 2
                if params.get('modality') == 'image':
                    self.task_heads[task] = ImageGenerationHead(
                        embed_dim=embed_dim,
                        img_size=params.get('img_size', 32),
                        channels=params.get('channels', 3),
                        seed=seed
                    )
                else: # this is shared across multiple text-generated tasks
                    self.task_heads[task] = SummarizationHead(
                        embed_dim=embed_dim,
                        vocab_size=params['vocab_size'],
                        max_length=params.get('max_length', 256),
                        use_lora=False,
                        seed=seed
                    )

    def forward(self, x, task=None):
        shared_features = x
        if task:
            return self.task_heads[task](shared_features)
        return {task: head(shared_features) for task, head in self.task_heads.items()}

# ======================== Final Client Model ============================
class MultiModalClientModel(nn.Module):
    def __init__(self, modalities, tasks, embed_dim=768, personal=False,
                 fusion=False, transformer_option='gated', 
                 domain_ids=None, task_domain_pair=None, seed=42, 
                 model_type='FLAVA', num_layers=1, num_heads=12, vocab_size=30522):
        super().__init__()
        self.modalities = modalities
        self.tasks = tasks
        self.personal = personal
        self.do_fusion = fusion
        self.transformer_option = transformer_option
        self.domain_ids = domain_ids
        
        # Encoders
        self.encoders = nn.ModuleDict()
        if 'image' in modalities:
            self.encoders['image'] = ImageEncoder(embed_dim, seed=seed, model_type=model_type)
        if 'text' in modalities:
            self.encoders['text'] = TextEncoder(embed_dim=embed_dim, use_lora=False, model_type=model_type, seed=seed, vocab_size=vocab_size)
        
        # Transformer
        self.fusion = GatedTransformer(
            embed_dim=embed_dim,
            modalities=modalities, 
            tasks=tasks, 
            domain_ids=domain_ids, 
            task_domain_pair=task_domain_pair,
            seed=seed,
            model_type=model_type,
            num_layers=num_layers,
            num_heads=num_heads
        )
        
        # Task head
        self.head = MultiTaskHead(embed_dim, tasks, seed=seed, model_type=model_type)

    def track_components(self, inputs, task):
        """Track which components are used during forward pass for a given task and return them"""
        components = {
            'encoders': set(),
            'mote_transformers': set(),
            'mome_transformers': set(),
            'mote_experts': set(),
            'mome_experts': set(),
            'shared_proj': False,
        }
        
        for mod in inputs.keys():
            if mod in self.encoders:
                components['encoders'].add(mod)
        components['shared_proj'] = True
        
        # Track transformer components used
        for mod in inputs.keys():
            if mod in self.fusion.mote_transformers:
                components['mote_transformers'].add(mod)
        components['mote_transformers'].add('shared')
        
        task_trans=task
        if task in self.fusion.mome_transformers:
            if 'image_classification' in task: 
                task_trans = 'image_classification'
            elif 'text_classification' in task:
                task_trans = 'text_classification'

            components['mome_transformers'].add(task_trans) 
            components['mome_transformers'].add('shared')
        
        for i in range(self.fusion.num_layers):
            for mod in list(inputs.keys()) + ['shared']:
                if mod in self.fusion.mote_experts[f'layer_{i}']:
                    if task_trans in self.fusion.mote_experts[f'layer_{i}'][mod]:
                        components['mote_experts'].add((f'layer_{i}', mod, task_trans))
                    if 'shared' in self.fusion.mote_experts[f'layer_{i}'][mod]:
                        components['mote_experts'].add((f'layer_{i}', mod, 'shared'))
            
            if task_trans in self.fusion.mome_experts[f'layer_{i}']:
                for mod in list(inputs.keys()) + ['shared']:
                    if mod in self.fusion.mome_experts[f'layer_{i}'][task_trans]:
                        components['mome_experts'].add((f'layer_{i}', task_trans, mod))
                if 'shared' in self.fusion.mome_experts[f'layer_{i}'][task_trans]:
                    components['mome_experts'].add((f'layer_{i}', 'shared', 'shared'))
                    for mod in list(inputs.keys()):
                        components['mome_experts'].add((f'layer_{i}', 'shared', mod))

        return components

    def replace_components(self, teacher_model, task, teacher_loss, student_loss, 
                       margin=0.1, all_components=None):
        """Replace components if needed"""
        components = all_components[task]
        
        replaced = False
        if (teacher_loss + margin) < student_loss:
            replaced = True

            # Replace encoders
            for mod in components['encoders']:
                # replacement for task-specific encoders
                self.encoders[mod].load_state_dict(teacher_model.encoders[mod].state_dict())
            
            # Replace shared_proj
            if components['shared_proj']:
                self.fusion.shared_proj.load_state_dict(teacher_model.fusion.shared_proj.state_dict())
            
            # Replace transformers
            for mod in components['mote_transformers']:
                # replacement for task-specific transformers
                self.fusion.mote_transformers[mod].load_state_dict(
                    teacher_model.fusion.mote_transformers[mod].state_dict())
            
            # Replace mome transformers
            for mome_task in components['mome_transformers']:  
                # replacement for task-specific transformers
                self.fusion.mome_transformers[mome_task].load_state_dict(
                    teacher_model.fusion.mome_transformers[mome_task].state_dict())
            
            # Replace experts
            for layer, mod, expert_task in components['mote_experts']:
                # Full replacement for task-specific experts
                self.fusion.mote_experts[layer][mod][expert_task].load_state_dict(
                    teacher_model.fusion.mote_experts[layer][mod][expert_task].state_dict())
            
            for layer, expert_task, mod in components['mome_experts']:
                self.fusion.mome_experts[layer][expert_task][mod].load_state_dict(
                    teacher_model.fusion.mome_experts[layer][expert_task][mod].state_dict())
            
            # Replace task head (always task-specific)
            self.head.task_heads[task].load_state_dict(
                teacher_model.head.task_heads[task].state_dict())

        if replaced:
            print(f'for task {task}, replacement has taken place!')
            return replaced
        return replaced

    def forward(self, inputs, domain_id, task=None, get_transformer=False, useDisentAFL=False):
        # Track components used for this task
        components = None
        if task is not None:
            components = self.track_components(inputs, task)
        
        encoded = {}
        keep_forward = True
        if 'image' in self.modalities and ("image" in task or self.do_fusion):
            encoded['image'] = self.encoders['image'](inputs['image'])
            keep_forward = True if self.do_fusion else False
        if 'text' in self.modalities and keep_forward and 'text' in task:
            encoded['text'] = self.encoders['text'](
                input_ids=inputs['text']['input_ids'],
                attention_mask=inputs['text']['attention_mask']
            )

        if useDisentAFL:
            features, latent_orthogonal = self.fusion(encoded, task, domain_id, useDisentAFL)
            return self.head(features, task), latent_orthogonal
        
        features = self.fusion(encoded, task, domain_id, False)
        
        if get_transformer:
            return features
        
        if self.training:
            return self.head(features, task), components
        else:
            return self.head(features, task)