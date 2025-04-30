import torch
import numpy as np
import matplotlib.pyplot as plt 
import time
import MinkowskiEngine as ME
import Train_Displacement as MM
import shutil
import os
from torch.utils.data import DataLoader
from functools import partial


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.input_files = sorted([f for f in os.listdir(os.path.join(data_folder, test_data_path)) if f.endswith('.npy')])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.data_folder, test_data_path, self.input_files[idx])
        coords_path = 'initial_npy_0.9.npy'
           
        input_data = np.load(input_path)
        input_data = np.array(input_data)
        
        coords = np.load(coords_path)
        coords = np.array(coords)

        return input_data, coords, self.input_files[idx]


def save_and_compare_image(predicted_array, file, prediction_path):

    # Adjust layout
    plt.tight_layout()
    base_name = file[:-4]
    directory_path_npy = prediction_path + '/' + base_name + ".npy"
    np.save(directory_path_npy, predicted_array)


def collation_fn(data_labels, voxel_size):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_data, coords, file_name = list(zip(*data_labels))  # Extracting file names
    input_data_batch, coords_batch = [], []

    coords_list = [coord / voxel_size for coord in coords]

    # Generate batched coordinates
    coords_batch = ME.utils.batched_coordinates(coords_list).to(device)

    # Concatenate all lists
    input_data_batch = torch.from_numpy(np.concatenate(input_data, 0)).float().to(device)

    return input_data_batch, coords_batch, file_name  # Returning file names along with data
    

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    
    model = MM.MinkUNet14(in_channels=3, out_channels=3, D=3).to(device)
    model.load_state_dict(torch.load('models/minkunet_model_MAE_displacement_final.pth')) 
    model.eval()
    
    if not os.path.exists("Test_Data/Prediction_Displacement_unseen"):
        os.makedirs("Test_Data/Prediction_Displacement_unseen")

    prediction_path = 'Test_Data/Prediction_Displacement_unseen'
    test_data_path = 'Test_Data/Input_Displacement_unseen'
    files = os.listdir(test_data_path)
    voxel_size = 0.11
    collation_fn_defined = partial(collation_fn, voxel_size=voxel_size)

    test_dataset = CustomDataset('./')
    test_loader = DataLoader(test_dataset, collate_fn=collation_fn_defined, batch_size=1, drop_last=False, shuffle=False)

    Error_X_sum = 0
    Error_Y_sum = 0
    Error_Z_sum = 0
    num_samples = 0
    
    for data in test_loader:

        input_data, coords, file_name = data
        
        start_time = time.time()
                
        sinput = ME.SparseTensor(input_data, coordinates = coords, quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
        outputs = model(sinput).slice(sinput).F.detach().cpu().numpy()
                
        end_time = time.time()
                
        #save_and_compare_image(outputs, file_name[0], prediction_path)

        torch.cuda.empty_cache()
                
                
                

            
            
