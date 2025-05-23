import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scipy.stats import entropy

# Define the path to your images
image_dir = r".\run\sample\ddpm_cifar10_pruned\process_0"

# Define CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Custom dataset to load images
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)
                            if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 images are 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset and dataloader
dataset = CustomImageDataset(image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=True)
model.eval()

# Predict classes
all_preds = []

with torch.no_grad():
    for images in dataloader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())

# Count frequency of each class
class_counts = Counter(all_preds)
frequencies = [class_counts.get(i, 0) for i in range(10)]


# Calculate entropy
probabilities = np.array(frequencies) / sum(frequencies)
diversity_score = entropy(probabilities, base=2)
print(f'Diversity Score (Entropy): {diversity_score:.4f}')

# Plot histogram
plt.figure(figsize=(10, 5))
plt.bar(classes, frequencies, color='skyblue')
plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.title(f'Diversity Score (Entropy): {diversity_score:.4f}')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Diversity_Score.png")
plt.show()



