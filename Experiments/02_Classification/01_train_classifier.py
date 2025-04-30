import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torch.utils import data
import os
from PIL import Image
import numpy as np
import cv2
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ImgAugTransform:
    def __init__(self, augmenter):
        self.augmenter = augmenter

    def __call__(self, img):
        img = np.array(img)
        img = self.augmenter(image=img)
        return F.to_pil_image(img)

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (128, 128)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

class Mydatasets(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root  # path of the document
        self.classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))] 
        self.imgs = [] # list to save path of the images
        self.labels = [] # list to save the labels (classes) of the images
        self.transform = transform

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root, class_name)
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

def main():
    set_seed(42)
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor()
    ])
    dataset = Mydatasets('tactile_data/sim_train', transform)
    print(dataset)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_db, val_db = torch.utils.data.random_split(dataset, [train_size, val_size]) 
    train_loader = torch.utils.data.DataLoader(train_db, batch_size=32, shuffle=True, num_workers=1)
    vali_loader = torch.utils.data.DataLoader(val_db, batch_size=8, shuffle=True, num_workers=1)

    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 12)  
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params, lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    average_loss = 0
    train_acc = 0
    batch_max= train_loader.dataset.__len__()/train_loader.batch_size
    batch_max=batch_max-batch_max%10
    
    epoches = 40
    val_losses = []
    
    for epoch in range(epoches):
        model.train()
        batch = 0

        for k, data in enumerate(train_loader, 1):
            # print('111')
            model.train()
            batch_images = data[0].to(device)
            batch_labels = data[1].to(device)
            out = model(batch_images)
            loss = criterion(out, batch_labels)
            average_loss = loss

            prediction = torch.max(out, 1)[1]
            train_correct = (prediction == batch_labels).sum()
            train_acc = (train_acc+(train_correct.float()) / train_loader.batch_size)/2

            optimizer.zero_grad() 
            loss.backward()  
            optimizer.step() 

            batch += 1

        model.eval()
        sum_validation = 0.0
        sum_valsample = 0.0
        for k, data_test in enumerate(vali_loader, 1):
            batch_images = data_test[0].to(device)
            batch_labels = data_test[1].to(device)
            out = model(batch_images)
            prediction = torch.max(out, 1)[1]
            vali_correct = (prediction == batch_labels).sum()
            sum_validation += vali_correct.float()
            sum_valsample += vali_loader.batch_size
        vali_acc = sum_validation / sum_valsample
        vali_acc = vali_acc.cpu().numpy()
        val_losses.append(vali_acc)
        print("Epoch: %d/10 || batch:%d/%d average_loss: %.3f || train_acc: %.2f || validation_acc: %.2f"
            % (epoch+1, batch, batch_max, average_loss, train_acc, vali_acc))

        if epoch == epoches - 1:
            model.eval()
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for inputs, labels in vali_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

            # Calculate confusion matrix
            cm = confusion_matrix(all_labels, all_preds)

            cm_df = pd.DataFrame(cm, index=dataset.classes, columns=dataset.classes)
            cm_df.to_csv('confusion_matrix.csv')  

            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=dataset.classes, yticklabels=dataset.classes)
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title("Confusion Matrix")
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            plt.tight_layout()
            plt.savefig('Confusion Matrix.png', dpi=600)
            plt.show()
        
    save_path = "final_model.pt"
    torch.save(model.state_dict(), save_path)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoches + 1), val_losses, label='Validation Loss', color='blue')
    plt.title('Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('validation_accuracy_curve.png', dpi=600)
    plt.show()

if __name__ == "__main__":
    main()
