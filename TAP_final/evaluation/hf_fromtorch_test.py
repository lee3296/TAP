# run script to turn torch datasets to hf

from datasets import Dataset
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

transform = transforms.Compose([
        transforms.Resize((64, 64)), #make 64x64 for consistency
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

fmnist_test = datasets.FashionMNIST(
    root='/path/to/HOME/TAP_final/data',
    train=False,
    download=True,
    transform=transform
)

def torch_dataset_to_hf(dataset):
    # Extract images and labels
    images = []
    labels = []
    
    for i in range(len(dataset)):
        img, label = dataset[i]
        
        # Convert tensor to PIL Image if it's a tensor
        if isinstance(img, torch.Tensor):
            # Remove batch dimension if present
            if img.dim() == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            
            # Convert to numpy and then to PIL
            img = img.numpy().transpose(1, 2, 0)  # CHW to HWC
            img = (img * 255).astype(np.uint8)  # Assuming img was normalized to [0,1]
            img = Image.fromarray(img)
        
        images.append(img)
        labels.append(label)
    
    # Create Hugging Face dataset
    hf_dataset = Dataset.from_dict({
        "image": images,
        "label": labels
    })
    
    return hf_dataset


fmnist_hf = torch_dataset_to_hf(fmnist_test)
fmnist_hf.save_to_disk("/path/to/HOME/TAP_final/data/fmnist_test")