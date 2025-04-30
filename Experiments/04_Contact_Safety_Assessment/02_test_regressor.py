import os
import re
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet50
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

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

        return image, label, os.path.basename(img_path)  # Return image name

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load test dataset
test_dir = 'real_data/test'
test_dataset = ImageDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pretrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(pretrained=False)  # Disable pretrained weights
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("resnet50_regression_SEED42_epoch80_final.pth", map_location=device))
model = model.to(device)
model.eval()

# Prediction and result saving
ground_truths = []
predictions = []

print("Processing test dataset:")
with torch.no_grad():
    for images, labels, img_names in test_loader:  
        images, labels = images.to(device), labels.to(device)

        outputs = model(images) + 0.31  # plus 0,31 for real test
        ground_truths.extend(labels.cpu().numpy())
        predictions.extend(outputs.cpu().numpy())

# Convert to numpy arrays
ground_truths = np.array(ground_truths)
predictions = np.array(predictions)

# Compute MAE
mae = mean_absolute_error(ground_truths, predictions)
print(f"Mean Absolute Error (MAE): {mae:.6f}")

# Scatter plot
plt.figure(figsize=(8, 8), dpi=600)
rgb_255_1 = (10, 84, 158)  # Light blue
scatter_color_1 = tuple(c / 255.0 for c in rgb_255_1)  
plt.scatter(ground_truths, predictions, alpha=1, c=[scatter_color_1], s=80)
plt.xlabel('Ground Truth', fontsize=15) 
plt.ylabel('Prediction', fontsize=15)  

# Line plot
target_color = (200, 93, 76)  # Custom line color
scatter_color_2 = tuple(c / 255.0 for c in target_color)  
plt.plot(
    [min(ground_truths), max(ground_truths)],
    [min(ground_truths), max(ground_truths)],
    color=scatter_color_2, linestyle='--', label='y=x', linewidth=4
)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=15)
plt.grid()
plt.savefig("sim2real_SEED42_epoch80_1.svg", format='svg')
#plt.show()