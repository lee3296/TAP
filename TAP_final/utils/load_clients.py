"""
Important notess:
- For image generation: 3x64x64 inputs and outputs
"""
from datasets import load_dataset, load_from_disk

#############################################
def load_client_datasets(num_groups=3, seed=42, model_type='FLAVA'): 
    """
    Args:
        num_groups: Number of client groups (each group has 10 clients with similar characteristics)
    """
    # Load the full datasets once
    fmnist_full = load_from_disk("/path/to/HOME/TAP_final/data/fmnist_hf") 
    
    # Initialize the datasets dictionary
    datasets_dict = {}
    
    # Calculate splits for all datasets that need to be divided across groups
    news_splits = split_range(120000, num_groups, 0)
    vqav2_splits = split_range(17000, num_groups, 0)
    fmnist_splits = split_range(len(fmnist_full), num_groups, 0)
    mmlu_law_splits = split_range(1534, num_groups, 0)
    mmlu_moral_splits = split_range(895, num_groups, 0)
    mmlu_medicine_splits = split_range(272, num_groups, 0)
    commongen_splits = split_range(67389, num_groups, 0)
    cifar100_splits = split_range(50000, num_groups, 0)
    imagenet_splits = split_range(100000, num_groups, 0)
    
    # For each group
    for group in range(num_groups):
        # Recalculate splits for this group
        news_splits = split_range(120000, num_groups, group)
        vqav2_splits = split_range(17000, num_groups, group)
        mmlu_law_splits = split_range(1534, num_groups, group)
        mmlu_moral_splits = split_range(895, num_groups, group)
        mmlu_medicine_splits = split_range(272, num_groups, group)
        fmnist_splits = split_range(len(fmnist_full), num_groups, group)
        commongen_splits = split_range(67389, num_groups, group)
        cifar100_splits = split_range(50000, num_groups, group)
        imagenet_splits = split_range(100000, num_groups, group)
        caltech_splits = split_range(24791, num_groups, group)

        if model_type == 'ViLT':
            fmnist_s = fmnist_splits[0]
            fmnist_e = fmnist_splits[1]
            fmnist_size = fmnist_e - fmnist_s
            fmnist_start = fmnist_s
            fmnist_end = fmnist_s + int(fmnist_size * 0.33)
            fmnist_start2 = fmnist_s + int(fmnist_size * 0.33)
            fmnist_end2 = fmnist_s + int(fmnist_size * 0.66)
            fmnist_start3 = fmnist_s + int(fmnist_size * 0.66)
            fmnist_end3 = fmnist_e

        # CommonGen
        common_s = commongen_splits[0]
        common_e = commongen_splits[1]
        common_size = common_e - common_s
        common_start = common_s
        common_end = common_s + int(common_size * 0.5)
        common_start2 = common_s + int(common_size * 0.5)
        common_end2 = common_e

        # Caltech-256
        caltech_s = caltech_splits[0]
        caltech_e = caltech_splits[1]
        caltech_size = caltech_e - caltech_s
        caltech_start = caltech_s
        caltech_end = caltech_s + int(caltech_size * 0.5)
        caltech_start2 = caltech_s + int(caltech_size * 0.5)
        caltech_end2 = caltech_e

        # AG News
        ag_size = news_splits[1] - news_splits[0]
        ag_start = news_splits[0]
        ag_end = news_splits[0] + int(ag_size * 0.33)
        ag_start2 = news_splits[0] + int(ag_size * 0.33)
        ag_end2 = news_splits[0] + int(ag_size * 0.66)
        ag_start3 = news_splits[0] + int(ag_size * 0.66)
        ag_end3 = news_splits[1]

        
        # For each client in the group (0-9)
        for client_offset in range(10):
            client_id = group * 10 + client_offset
            
            # Client 0 pattern 
            if client_offset == 0:
                if model_type == 'FLAVA':
                    datasets_dict[client_id] = { 
                        'modalities': ['image'],
                        'tasks': ['imagenet_image_classification'],
                        'datasets': {
                            'imagenet_image_classification': (
                                load_dataset("zh-plus/tiny-imagenet", split='train')
                                .shuffle(seed=seed)
                                .select(range(imagenet_splits[0], imagenet_splits[1])),
                                'imagenet_image_classification',
                                ['image'],
                                'finegrain_general_object'
                            )
                        },
                        'fusion': False,
                        'dataset_names': {'imagenet_image_classification': 'zh-plus/tiny-imagenet'}, 
                        'modality_task_pairs': {'imagenet_image_classification': ['image']}
                    }
                elif model_type == 'ViLT':
                    datasets_dict[client_id] = { 
                        'modalities': ['image'],
                        'tasks': ['image_generation'],
                        'datasets': {
                            'image_generation': (
                                fmnist_full.select(range(fmnist_start, fmnist_end)),
                                'image_generation', 
                                ['image'],
                                'general_object'
                            )
                        },
                        'fusion': False,
                        'dataset_names': {'image_generation': 'fmnist'}, 
                        'modality_task_pairs': {'image_generation': ['image']}
                    }

            # Client 1 & 2 pattern
            elif client_offset in [1, 2]:
                if client_offset == 1:
                    client_start = ag_start
                    client_end = ag_end
                else:
                    client_start = ag_start2
                    client_end = ag_end2
                datasets_dict[client_id] = { 
                    'modalities': ['text'],
                    'tasks': ['agnews_text_classification'],
                    'datasets': {
                        'agnews_text_classification': (
                            load_dataset("sh0416/ag_news", split='train')
                            .shuffle(seed=seed)
                            .select(range(client_start, client_end)),
                            'agnews_text_classification',
                            ['text'],
                            'news'
                        )
                    },
                    'fusion': False,
                    'dataset_names': {'agnews_text_classification': "sh0416/ag_news"},
                    'modality_task_pairs': {'agnews_text_classification': ['text']}
                }
            
            # Client 3 pattern
            elif client_offset == 3:
                datasets_dict[client_id] = { 
                    'modalities': ['image'],
                    'tasks': ['image_generation'],
                    'datasets': {
                        'image_generation': (
                            load_dataset("ilee0022/Caltech-256", split='train').
                            shuffle(seed=seed).
                            select(range(caltech_start, caltech_end)),
                            'image_generation', 
                            ['image'],
                            'general_object'
                        ),
                    },
                    'fusion': False,
                    'dataset_names': {'image_generation': "ilee0022/Caltech-256"},
                    'modality_task_pairs': {'image_generation': ['image']}
                }
            
            # Client 4 & 5 pattern
            elif client_offset in [4, 5]:
                # Split VQA data 50-50 within group
                group_vqa_start = vqav2_splits[0]
                group_vqa_end = vqav2_splits[1]
                group_vqa_size = group_vqa_end - group_vqa_start
                
                if client_offset == 4:
                    client_vqa_start = group_vqa_start
                    client_vqa_end = group_vqa_start + int(group_vqa_size * 0.5)
                else:
                    client_vqa_start = group_vqa_start + int(group_vqa_size * 0.5)
                    client_vqa_end = group_vqa_end
                
                datasets_dict[client_id] = { 
                    'modalities': ['image', 'text'],
                    'tasks': ['text_generation'], 
                    'datasets': {
                        'text_answering': (
                            load_dataset("merve/vqav2-small")['validation']
                                .shuffle(seed=seed)
                                .select(range(client_vqa_start, client_vqa_end)),
                            'text_generation', 
                            ['image', 'text'],
                            'visual_answering'
                        )
                    },
                    'fusion': True,
                    'fusion-groups': ['image-text'],
                    'dataset_names': {'text_generation': "merve/vqav2-small"},
                    'modality_task_pairs': {'text_generation': ['image', 'text']}
                }
            
            # Client 6 pattern
            elif client_offset == 6:
                datasets_dict[client_id] = { 
                    'modalities': ['text'],
                    'tasks': ['text_generation'],
                    'datasets': {
                        'law_answering': (load_dataset("cais/mmlu", "professional_law")['test']
                                          .shuffle(seed=seed)
                                          .select(range(mmlu_law_splits[0], mmlu_law_splits[1])),
                                        'text_generation', ['text'], 'law_answering'),
                        'medicine_answering': (load_dataset("cais/mmlu", "professional_medicine")['test']
                                               .shuffle(seed=seed)
                                               .select(range(mmlu_medicine_splits[0], mmlu_medicine_splits[1])),
                                               'text_generation', ['text'], 'medicine_answering'),
                        'moral_answering': (load_dataset("cais/mmlu", "moral_scenarios")['test']
                                            .shuffle(seed=seed)
                                            .select(range(mmlu_moral_splits[0], mmlu_moral_splits[1])),
                                            'text_generation', ['text'], 'moral_answering'),
                    },
                    'fusion': False,
                    'dataset_names': {'text_generation': 'cais/mmlu'},
                    'modality_task_pairs': {'law_answering': ['text'], 'medicine_answering': ['text'], 'moral_answering': ['text']}
                }
            
            # Client 7 pattern
            elif client_offset == 7:
                if model_type == 'FLAVA':
                    datasets_dict[client_id] = { 
                        'modalities': ['image', 'text'],
                        'tasks': ['cifar100_image_classification', 'agnews_text_classification'],
                        'datasets': {
                            'cifar100_image_classification': (
                                load_dataset("uoft-cs/cifar100", split='train').shuffle(seed=seed)
                                .select(range(cifar100_splits[0], cifar100_splits[1])),
                                'cifar100_image_classification', 
                                ['image'],
                                'finegrain_general_object'
                            ),
                            'agnews_text_classification': (
                                load_dataset("sh0416/ag_news", split='train')
                                .shuffle(seed=seed)
                                .select(range(ag_start3, ag_end3)),
                                'agnews_text_classification',
                                ['text'],
                                'news'
                            )
                        },
                        'fusion': False,
                        'dataset_names': {
                            'cifar100_image_classification': "uoft-cs/cifar100", 
                            'agnews_text_classification': "sh0416/ag_news"
                        },
                        'modality_task_pairs': {
                            'cifar100_image_classification': ['image'], 
                            'agnews_text_classification': ['text']
                        }
                    }
                elif model_type == 'ViLT':
                    datasets_dict[client_id] = { 
                        'modalities': ['image', 'text'],
                        'tasks': ['image_generation', 'agnews_text_classification'],
                        'datasets': {
                            'image_generation': (
                                fmnist_full.select(range(fmnist_start2, fmnist_end2)),
                                'image_generation', 
                                ['image'],
                                'general_object'
                            ),
                            'agnews_text_classification': (
                                load_dataset("sh0416/ag_news", split='train')
                                .shuffle(seed=seed)
                                .select(range(ag_start3, ag_end3)),
                                'agnews_text_classification',
                                ['text'],
                                'news'
                            )
                        },
                        'fusion': False,
                        'dataset_names': {
                            'image_generation': "fmnist", 
                            'agnews_text_classification': "sh0416/ag_news"
                        },
                        'modality_task_pairs': {
                            'image_generation': ['image'], 
                            'agnews_text_classification': ['text']
                        }
                    }
            
            # Client 8 pattern
            elif client_offset == 8:
                if model_type == 'FLAVA':
                    datasets_dict[client_id] = { 
                        'modalities': ['text', 'image'],
                        'tasks': ['text_generation', 'image_generation'],
                        'datasets': {
                            'common_sense_answering': (load_dataset("allenai/common_gen", split='train').shuffle(seed=seed)
                                                .select(range(common_start, common_end)),
                                                'text_generation', ['text'], 'common_sense_answering'),
                            'image_generation': (
                                fmnist_full.select(range(fmnist_splits[0], fmnist_splits[1])),
                                'image_generation', 
                                ['image'],
                                'general_object'
                            )
                        },
                        'fusion': False,
                        'dataset_names': {'text_generation': "allenai/common_gen", 'image_generation': "fmnist"},
                        'modality_task_pairs': {
                            'common_sense_answering': ['text'], 
                            'image_generation': ['image']
                        }
                    }
                elif model_type == 'ViLT':
                    datasets_dict[client_id] = { 
                        'modalities': ['text', 'image'],
                        'tasks': ['text_generation', 'image_generation'],
                        'datasets': {
                            'common_sense_answering': (load_dataset("allenai/common_gen", split='train').shuffle(seed=seed)
                                                .select(range(common_start, common_end)),
                                                'text_generation', ['text'], 'common_sense_answering'),
                            'image_generation': (
                                fmnist_full.select(range(fmnist_start3, fmnist_end3)),
                                'image_generation', 
                                ['image'],
                                'general_object'
                            )
                        },
                        'fusion': False,
                        'dataset_names': {'text_generation': "allenai/common_gen", 'image_generation': "fmnist"},
                        'modality_task_pairs': {
                            'common_sense_answering': ['text'], 
                            'image_generation': ['image']
                        }
                    }
            
            # Client 9 pattern
            elif client_offset == 9:
                datasets_dict[client_id] = { 
                    'modalities': ['text', 'image'],
                    'tasks': ['text_generation', 'image_generation'],
                    'datasets': {
                        'common_sense_answering': (load_dataset("allenai/common_gen", split='train').shuffle(seed=seed)
                                            .select(range(common_start2, common_end2)),
                                            'text_generation', ['text'], 'common_sense_answering'),
                        'image_generation': (
                            load_dataset("ilee0022/Caltech-256", split='train').
                            shuffle(seed=seed).
                            select(range(caltech_start2, caltech_end2)),
                            'image_generation', 
                            ['image'], 
                            'general_object'
                        )
                    },
                    'fusion': False,
                    'dataset_names': {'text_generation': "allenai/common_gen", 'image_generation': "ilee0022/Caltech-256"},
                    'modality_task_pairs': {
                        'common_sense_answering': ['text'],
                        'image_generation': ['image']
                    }
                }
    return datasets_dict

def split_range(total_size, num_groups, group_num):
    """Helper function to calculate start and end indices for a group's share of a dataset."""
    per_group = total_size // num_groups
    start = group_num * per_group
    end = (group_num + 1) * per_group if group_num != num_groups - 1 else total_size
    return (start, end)