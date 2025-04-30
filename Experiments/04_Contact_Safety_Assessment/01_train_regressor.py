import os
import re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

# Set a fixed random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
        self.labels = [self.extract_label(f) for f in os.listdir(image_dir) if f.endswith('.png')]

    def extract_label(self, filename):
        match = re.search(r'score([-+]?[0-9]*\.?[0-9]+)', filename)
        if match:
            return float(match.group(1))
        else:
            raise ValueError(f"Invalid filename format: {filename}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset
image_dir = 'sim_data/train'
dataset = ImageDataset(image_dir, transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pretrained ResNet-50 model
model = resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training parameters
num_epochs = 80
train_losses, val_losses = [], []

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "resnet50_regression.pth")

# Save training logs to CSV
results_df = pd.DataFrame({
    'Epoch': list(range(1, num_epochs + 1)),
    'Train Loss': train_losses,
    'Val Loss': val_losses
})
results_df.to_csv("training_results.csv", index=False)

# Plot loss curve
plt.figure(figsize=(10, 5))
plt.plot(results_df['Epoch'], results_df['Train Loss'], label='Train Loss')
plt.plot(results_df['Epoch'], results_df['Val Loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("Training_loss_curve.png")
plt.show()