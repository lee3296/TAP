from datasets import load_dataset, load_from_disk

def load_test_datasets(model_type='FLAVA'):
    fmnist_hf = load_from_disk("/path/to/HOME/TAP_final/data/fmnist_test")
    
    # Initialize the datasets dictionary
    datasets_dict = {}
    for group in range(1):
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
                                load_dataset("zh-plus/tiny-imagenet", split='valid'),
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
                                fmnist_hf,
                                'image_generation', 
                                ['image'],
                                'general_object'
                            )
                        },
                        'fusion': False,
                        'dataset_names': {'image_generation': "fmnist"},
                        'modality_task_pairs': {
                            'image_generation': ['image']
                        }
                    }

            
            elif client_offset in [1,2]: 
                datasets_dict[client_id] = { 
                    'modalities': ['text'],
                    'tasks': ['agnews_text_classification'],
                    'datasets': {
                        'agnews_text_classification': (
                            load_dataset("sh0416/ag_news", split='test'),
                            'agnews_text_classification',
                            ['text'],
                            'news'
                        ),
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
                            load_dataset("ilee0022/Caltech-256", split='test'),
                            'image_generation', 
                            ['image'],
                            'general_object'
                        )
                    },
                    'fusion': False,
                    'dataset_names': {'image_generation': "ilee0022/Caltech-256"},
                    'modality_task_pairs': {'image_generation': ['image']}
                }
            
            # Client 4 & 5 pattern 
            elif client_offset in [4, 5]:
                datasets_dict[client_id] = { 
                    'modalities': ['image', 'text'],
                    'tasks': ['text_generation'], 
                    'datasets': {
                        'text_answering': (
                            load_dataset("merve/vqav2-small")['validation']
                                .shuffle(seed=42)
                                .select(range(17000, 21435)),
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
                        'law_answering': (load_dataset("cais/mmlu", "professional_law")['validation'], 'text_generation', ['text'], 'law_answering'),
                        'medicine_answering': (load_dataset("cais/mmlu", "professional_medicine")['validation'], 'text_generation', ['text'], 'medicine_answering'),
                        'moral_answering': (load_dataset("cais/mmlu", "moral_scenarios")['validation'], 'text_generation', ['text'],'moral_answering'),
                    },
                    'fusion': False,
                    'dataset_names': {'text_generation': 'cais/mmlu'},
                    'modality_task_pairs': {'law_answering': ['text'], 'medicine_answering': ['text'], 'moral_answering': ['text']}
                }
            
            # Client 7 pattern 
            elif client_offset == 7:
                # Now use the structured Hugging Face Dataset object
                if model_type == 'FLAVA':
                    datasets_dict[client_id] = { 
                        'modalities': ['image', 'text'],
                        'tasks': ['cifar100_image_classification', 'agnews_text_classification'],
                        'datasets': {
                            'cifar100_image_classification': (
                                load_dataset("uoft-cs/cifar100", split='test'),
                                'cifar100_image_classification', 
                                ['image'],
                                'finegrain_general_object'
                            ),
                            'agnews_text_classification': (
                                load_dataset("sh0416/ag_news", split='test'),
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
                                fmnist_hf,
                                'image_generation', 
                                ['image'],
                                'general_object'
                            ),
                            'agnews_text_classification': (
                                load_dataset("sh0416/ag_news", split='test'),
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
                datasets_dict[client_id] = { 
                    'modalities': ['text', 'image'],
                    'tasks': ['text_generation', 'image_generation'],
                    'datasets': {
                       'common_sense_answering': (load_dataset("allenai/common_gen", split='validation'),
                                            'text_generation', ['text'], 'common_sense_answering'),
                        'image_generation': (
                            fmnist_hf,
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
                        'common_sense_answering': (load_dataset("allenai/common_gen", split='validation'),
                                            'text_generation', ['text'], 'common_sense_answering'),
                        'image_generation': (
                            load_dataset("ilee0022/Caltech-256", split='test'),
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