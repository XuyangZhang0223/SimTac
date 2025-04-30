import torch
import numpy as np
import matplotlib.pyplot as plt 
import time
import MinkowskiEngine as ME
import libs.Scripts.STN_structure as MM

def CNN_predict_Displacement(file):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    voxel_size = 0.11
    input_data = np.array(file)
    input_data = torch.from_numpy(input_data).to(device)
    coords_path = './libs/Scripts/initial_npy_0.9.npy'
    coords = np.load(coords_path)
    coords = np.array(coords)
    coords = ME.utils.batched_coordinates([coords / voxel_size], dtype=torch.float32).to(device)
    model = MM.MinkUNet14(in_channels=3, out_channels=3, D=3).to(device)
    model.load_state_dict(torch.load('./libs/Scripts/STN_model_displacement.pth')) 
    model.eval()  
    input_data = input_data.to(coords.dtype).to(coords.device)                
    sinput = ME.SparseTensor(input_data, coordinates = coords, quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
    outputs = model(sinput).slice(sinput).F.detach().cpu().numpy()

    return outputs


def CNN_predict_Force(file):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    voxel_size = 0.11
    input_data = np.array(file)
    input_data = torch.from_numpy(input_data).to(device)
    coords_path = './libs/Scripts/initial_npy_0.9.npy'
    coords = np.load(coords_path)
    coords = np.array(coords)
    coords = ME.utils.batched_coordinates([coords / voxel_size], dtype=torch.float32).to(device)
    model = MM.MinkUNet14(in_channels=3, out_channels=3, D=3).to(device)
    model.load_state_dict(torch.load('./libs/Scripts/STN_model_force.pth')) 
    model.eval()  
    input_data = input_data.to(coords.dtype).to(coords.device)                    
    sinput = ME.SparseTensor(input_data, coordinates = coords, quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
    outputs = model(sinput).slice(sinput).F.detach().cpu().numpy()
            
    return outputs

                
                

            
            
