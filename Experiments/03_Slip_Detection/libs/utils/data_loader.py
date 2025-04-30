import os
import re
import random
import cv2
import torch
from torch.utils import data
from natsort import natsorted

class Tactile_dataset(data.Dataset):
    def __init__(self, data_path='./data'):
        self.data_path = data_path
        self.label_files = []
        self.train_data = []
        self.image_paths = []  # To store image paths
        for root, dirs, files in os.walk(data_path, topdown=True):
            for file in files:
                if file.endswith('.dat'):
                    self.label_files.append(os.path.join(root, file))
                    
        pat = re.compile(r'object(\d+)_result')
        
        for label_file in self.label_files:
            idx = pat.search(label_file).group(1)
            fp = open(label_file, 'r')
            lines = fp.readlines()
            
            self.train_data.extend([line.replace('\n','') + ' ' + idx for line in lines])

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        train_data = self.train_data[index]
        output_tacitle_imgs = []
        image_paths = []  # To store paths for each sample

        train_data = train_data.split(' ')
        object_id = train_data[-1]
        test_status_id = train_data[0]
        status = int(train_data[1]) # Label
        label = torch.tensor([status]).long()
        label = torch.squeeze(label)

        path = os.path.join(self.data_path, 'object' + object_id, test_status_id)
        
        tacitle_img_paths = []
        for root, dirs, files in os.walk(path, topdown=True):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    tacitle_img_paths.append(os.path.join(root, file))

        tacitle_img_paths = natsorted(tacitle_img_paths)

        if len(tacitle_img_paths) >= 8:
            selected_paths = (
                random.sample(tacitle_img_paths[-8:-7], 1) + 
                random.sample(tacitle_img_paths[-7:-6], 1) +
                random.sample(tacitle_img_paths[-6:-5], 1) +
                random.sample(tacitle_img_paths[-5:-4], 1) +
                random.sample(tacitle_img_paths[-4:-3], 1) +
                random.sample(tacitle_img_paths[-3:-2], 1) +
                random.sample(tacitle_img_paths[-2:-1], 1) +
                random.sample(tacitle_img_paths[:-1], 1))

            selected_paths = natsorted(selected_paths) 

            for tacitle_img_path in selected_paths:
                tacitle_img = cv2.imread(tacitle_img_path)
                size = tacitle_img.shape
                tacitle_img_tensor = torch.from_numpy(tacitle_img.reshape(size[2], size[0], size[1])).float()/ 255.0
                output_tacitle_imgs.append(tacitle_img_tensor)
                image_paths.append(tacitle_img_path)  # Store the image path
                # cv2.imshow("Input to Network", tacitle_img)
                # cv2.waitKey(300)

        torch.cuda.empty_cache()

        return output_tacitle_imgs, label  # Return image paths along with images and labels image_paths


