import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import MinkowskiEngine as ME
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os
## add torch dataloader
from torch.utils.data import DataLoader, random_split
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from examples.resnet import ResNetBase
from functools import partial
import csv
import os



## model structure
class MinkUNetBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        #self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        #self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        #self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        #self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        #self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        #self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        #self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        #self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        #self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
        #self.relu = ME.MinkowskiReLU(inplace=True)
        self.relu = ME.MinkowskiLeakyReLU()

    def forward(self, x):
    
        #print("x shape:", x.shape)
    
        out = self.conv0p1s1(x)
        #out = self.bn0(out)
        out_p1 = self.relu(out)
        
        #print("Final output shape:", out.shape)

        out = self.conv1p1s2(out_p1)
        #out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)
        
        #print("Final output shape:", out.shape)

        out = self.conv2p2s2(out_b1p2)
        #out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)
        
        #print("Final output shape:", out.shape)

        out = self.conv3p4s2(out_b2p4)
        #out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)
        
        #print("Final output shape:", out.shape)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        #out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)
        
        #print("Final output shape:", out.shape)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        #out = self.bntr4(out)
        out = self.relu(out)
        
        #print("Final output shape:", out.shape)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)
        
        #print("Final output shape:", out.shape)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        #out = self.bntr5(out)
        out = self.relu(out)
        
        #print("Final output shape:", out.shape)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)
        
        #print("Final output shape:", out.shape)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        #out = self.bntr6(out)
        out = self.relu(out)
        
        #print("Final output shape:", out.shape)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)
        
        #print("Final output shape:", out.shape)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        #out = self.bntr7(out)
        out = self.relu(out)
        
        #print("Final output shape:", out.shape)

        out = ME.cat(out, out_p1)
        out = self.block8(out)
        
        #print("Final output shape:", out.shape)

        return self.final(out)


class MinkUNet14(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class MinkUNet18(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class MinkUNet34(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet50(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet101(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class MinkUNet14A(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)

class MinkUNet14E(MinkUNet14):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class MinkUNet14B(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet14C(MinkUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class MinkUNet14D(MinkUNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet18A(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet18B(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet18D(MinkUNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet34A(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class MinkUNet34B(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class MinkUNet34C(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

## dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.input_files = sorted([f for f in os.listdir(os.path.join(data_folder, 'Fine_tune_Data/MidHigh_Youngs_Modulus/Input')) if f.endswith('.npy')])
        self.gt_files = sorted([f for f in os.listdir(os.path.join(data_folder, 'Fine_tune_Data/MidHigh_Youngs_Modulus/Ground_Truth_Force')) if f.endswith('.npy')])
        assert len(self.input_files) == len(self.gt_files), "Number of input files must match number of ground truth files."

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.data_folder, 'Fine_tune_Data/MidHigh_Youngs_Modulus/Input', self.input_files[idx])
        gt_path = os.path.join(self.data_folder, 'Fine_tune_Data/MidHigh_Youngs_Modulus/Ground_Truth_Force', self.gt_files[idx])
        coords_path = 'initial_npy_0.9.npy'
           
        input_data = np.load(input_path)
        input_data = np.array(input_data)
        #print('input_data.shape: ', input_data.shape)
        
        gt_data = np.load(gt_path)
        gt_data = np.array(gt_data)
        gt_data*= 50
        
        #print('gt_data.shape: ', gt_data.shape)

        coords = np.load(coords_path)
        coords = np.array(coords)
        #print('coords.shape: ', coords.shape)

        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #input_tensor = torch.from_numpy(input_data).float()
        #gt_tensor = torch.from_numpy(gt_data).float()
        #coords = torch.from_numpy(coords).float()
        
        #input_tensor = ME.SparseTensor(input_tensor, coordinates=coords, device=device)
        #gt_tensor = ME.SparseTensor(gt_tensor, coordinates=coords, device=device)

        #return input_tensor, gt_tensor, coords
        return input_data, gt_data, coords

    
def collation_fn(data_labels, voxel_size):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_data, gt_data, coords = list(zip(*data_labels))
    input_data_batch, gt_data_batch, coords_batch = [], [], []    
    
    
    coords_list = [coord / voxel_size for coord in coords]

    # Generate batched coordinates
    coords_batch = ME.utils.batched_coordinates(coords_list).to(device)
    #coords_batch = ME.utils.batched_coordinates(coords_list, dtype=torch.float32).to(device)


    # Concatenate all lists
    input_data_batch = torch.from_numpy(np.concatenate(input_data, 0)).float().to(device)
    gt_data_batch = torch.from_numpy(np.concatenate(gt_data, 0)).float().to(device)
    
    #print('input_data_batch.shape: ', input_data_batch.shape)
    #print('gt_data_batch.shape: ', gt_data_batch.shape)
    #print('coords_batch.shape: ', coords_batch.shape)

    return input_data_batch, gt_data_batch, coords_batch


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 32
    num_epochs = 80  # Set to a lower number since it's fine-tuning
    fine_tune_learning_rate = 1e-3  # Typically lower for fine-tuning
    voxel_size = 0.11

    # Define collation function
    collation_fn_defined = partial(collation_fn, voxel_size=voxel_size)

    # Load your new dataset for fine-tuning
    fine_tune_dataset = CustomDataset('./')  # Assuming new dataset path
    dataset_size = len(fine_tune_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(fine_tune_dataset, [train_size, val_size])

    # Define data loaders
    train_loader = DataLoader(train_dataset, collate_fn=collation_fn_defined, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, collate_fn=collation_fn_defined, batch_size=batch_size, drop_last=False, shuffle=False)

    # Loss function
    criterion = nn.L1Loss()

    # Load the model
    net = MinkUNet14(in_channels=3, out_channels=3, D=3)
    pre_trained_model_path = 'models/minkunet_model_MAE_force_final.pth'
    net.load_state_dict(torch.load(pre_trained_model_path))

    # code to freeze the encoder
    #for name, param in net.named_parameters():
        #if name.startswith(('conv0p1s1', 'conv1p1s2', 'block1', 'conv2p2s2', 'block2', 'conv3p4s2', 'block3', 'conv4p8s2', 'block4')):
            #param.requires_grad = False  

    #print("Frozen layers:")
    #for name, param in net.named_parameters():
        #if not param.requires_grad:
            #print(name)

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    # Define optimizer with a smaller learning rate
    optimizer = Adam(net.parameters(), lr=fine_tune_learning_rate)

    # Prepare CSV file for logging
    log_file = 'fine_tuning_losses_Force_MidHigh_E.csv'
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss'])

    # Fine-tuning process
    val_losses = []
    
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            input_data, gt_data, coords = data
            sinput = ME.SparseTensor(input_data, coordinates=coords, quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
            outputs = net(sinput).slice(sinput).F

            optimizer.zero_grad()
            loss = criterion(outputs, gt_data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.6f}')

        # Validation step
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                input_data, gt_data, coords = data
                sinput = ME.SparseTensor(input_data, coords)
                outputs = net(sinput).slice(sinput).F
                loss = criterion(outputs, gt_data) / 50
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.6f}')
        
        # Log results to CSV
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])

    # Plot validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='blue')
    plt.title('Validation Loss Curves (Fine-tuning)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    #plt.savefig('fine_tuning_losses_Force_MidHigh_E.png')
    #plt.show()

    # Save the fine-tuned model
    torch.save(net.state_dict(), 'models_fine_tune/fine_tuned_minkunet_model_Force_MidHigh_E.pth')
