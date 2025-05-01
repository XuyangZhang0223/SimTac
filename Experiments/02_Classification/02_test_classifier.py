import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
import imgaug.augmenters as iaa
from PIL import Image
import os
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import json

class ImgAugTransform:
    def __init__(self, augmenter):
        self.augmenter = augmenter

    def __call__(self, img):
        img = np.array(img)
        img = self.augmenter(image=img)
        return F.to_pil_image(img)

class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, root, class_list_path="class_names.json", transform=None):
        self.root = root
        with open(class_list_path, "r") as f:
            self.classes = json.load(f)  
        self.imgs = []
        self.labels = []
        self.transform = transform
        print('class order (from json): ', self.classes)

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root, class_name)
            if not os.path.exists(class_dir): continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.imgs.append(img_path)
                self.labels.append(class_idx)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Testing Progress", unit="batch")):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print('Test Accuracy: {:.2f}%'.format(accuracy))

    cm = confusion_matrix(all_labels, all_preds)
    class_names = test_loader.dataset.classes
    class_names = [name.replace("Double_cylinder_side", "Double_side") for name in class_names]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 17},
        cbar_kws={"shrink": 1, "format": "%d"},
        cbar=False
    )
    plt.xlabel("Predicted Labels", fontsize=16)
    plt.ylabel("True Labels", fontsize=16)
    plt.title("Confusion Matrix", fontsize=18)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(rotation=45, fontsize=14)
    plt.tight_layout()
    plt.savefig('Confusion_Matrix_sim2sim.svg')
    plt.show()

def main():
    saved_model_path = "final_model.pt"
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 12)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(saved_model_path, map_location=device))
    model.to(device)

    transform = T.Compose([
        T.Resize(224),
        T.ToTensor()
    ])

    test_dataset = Mydatasets('tactile_data/sim_test', class_list_path="class_names.json", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()
