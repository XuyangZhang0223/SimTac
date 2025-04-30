import torch
import numpy as np
import matplotlib.pyplot as plt 
import time
import MinkowskiEngine as ME
import Scrips.Train_Displacement as MM

def CNN_predict_Displacement(file):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    voxel_size = 0.11

    input_data = np.array(file)
    input_data = torch.from_numpy(input_data).to(device)

    coords_path = 'Scrips/initial_npy.npy'
    coords = np.load(coords_path)
    coords = np.array(coords)
    coords = ME.utils.batched_coordinates([coords / voxel_size], dtype=torch.float32).to(device)
    
    model = MM.MinkUNet14(in_channels=3, out_channels=3, D=3).to(device)
    model.load_state_dict(torch.load('STN/minkunet_model_MAE_displacement_final_gelsight_0.11.pth')) 
    model.eval()  
    
    start_time = time.time()

    input_data = input_data.to(coords.dtype).to(coords.device)
                        
    sinput = ME.SparseTensor(input_data, coordinates = coords, quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
    outputs = model(sinput).slice(sinput).F.detach().cpu().numpy()
            
    end_time = time.time()               
            
    # print("Predicted npy saved!")
    
    prediction_time = end_time - start_time

    # print("Predictions:")
    # print(f"Displacement Prediction time: {prediction_time} seconds")
    # print('-----------------------------------------------------')

    return outputs


def CNN_predict_Force(file):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    voxel_size = 0.11

    input_data = np.array(file)
    input_data = torch.from_numpy(input_data).to(device)

    #coords_path = 'initial_npy.npy'
    coords_path = 'Scrips/initial_npy.npy'
    coords = np.load(coords_path)
    coords = np.array(coords)
    coords = ME.utils.batched_coordinates([coords / voxel_size], dtype=torch.float32).to(device)
    
    model = MM.MinkUNet14(in_channels=3, out_channels=3, D=3).to(device)
    model.load_state_dict(torch.load('STN/minkunet_model_MAE_force_final_gelsight_0.11.pth')) 
    model.eval()  
    
    start_time = time.time()

    input_data = input_data.to(coords.dtype).to(coords.device)
                        
    sinput = ME.SparseTensor(input_data, coordinates = coords, quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
    outputs = model(sinput).slice(sinput).F.detach().cpu().numpy()
            
    end_time = time.time()               
            
    # print("Predicted npy saved!")
    
    prediction_time = end_time - start_time

    # print("Predictions:")
    # print(f"Force Prediction time: {prediction_time} seconds")
    # print('-----------------------------------------------------')

    return outputs

                
                

            
            
