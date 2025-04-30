import os
os.environ['OMP_NUM_THREADS'] = '12'
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import numpy as np
import matplotlib.pyplot as plt

## add torch dataloader
from torch.utils.data import DataLoader, random_split

import MinkowskiEngine as ME

from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from libs.Scripts.resnet import ResNetBase
from functools import partial
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
        self.input_files = sorted([f for f in os.listdir(os.path.join(data_folder, 'Train/Dynamic_Input_Train_Force')) if f.endswith('.npy')])
        self.gt_files = sorted([f for f in os.listdir(os.path.join(data_folder, 'Train/Static_Ground_Truth_Train_Force')) if f.endswith('.npy')])
        assert len(self.input_files) == len(self.gt_files), "Number of input files must match number of ground truth files."

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.data_folder, 'Train/Dynamic_Input_Train_Force', self.input_files[idx])
        gt_path = os.path.join(self.data_folder, 'Train/Static_Ground_Truth_Train_Force', self.gt_files[idx])
        coords_path = 'initial_npy.npy'       
        input_data = np.load(input_path)
        input_data = np.array(input_data)
        gt_data = np.load(gt_path)
        gt_data = np.array(gt_data)
        gt_data*= 50

        coords = np.load(coords_path)
        coords = np.array(coords)
        
        return input_data, gt_data, coords

def collation_fn(data_labels, voxel_size):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_data, gt_data, coords = list(zip(*data_labels))
    input_data_batch, gt_data_batch, coords_batch = [], [], []    
    coords_list = [coord / voxel_size for coord in coords]
    coords_batch = ME.utils.batched_coordinates(coords_list).to(device)
    input_data_batch = torch.from_numpy(np.concatenate(input_data, 0)).float().to(device)
    gt_data_batch = torch.from_numpy(np.concatenate(gt_data, 0)).float().to(device)

    return input_data_batch, gt_data_batch, coords_batch


if __name__ == '__main__':

    batch_size = 2
    num_epochs = 40
    learning_rate = 1e-3
    voxel_size = 0.05
    collation_fn_defined = partial(collation_fn, voxel_size=voxel_size)
    dataset = CustomDataset('./')
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, collate_fn=collation_fn_defined, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, collate_fn=collation_fn_defined, batch_size=1, drop_last=False, shuffle=False)
    criterion = nn.L1Loss()
    net = MinkUNet14(in_channels=3, out_channels=3, D=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    optimizer = Adam(net.parameters(), lr=learning_rate)
    val_losses = []

    for epoch in range(num_epochs):
        train_iter = iter(train_loader)
        net.train()
        for i, data in enumerate(train_iter):
            input_data, gt_data, coords = data
            sinput = ME.SparseTensor(input_data, coordinates = coords, quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
            outputs = net(sinput).slice(sinput).F
            optimizer.zero_grad()
            loss = criterion(outputs, gt_data)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            learning_rate = learning_rate/10
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                input_data, gt_data, coords = data   
                sinput = ME.SparseTensor(input_data, coords)
                outputs = net(sinput).slice(sinput).F     
                loss = criterion(outputs, gt_data)
                loss = loss/50
                val_loss += loss.item()
        average_val_loss = val_loss / len(val_loader)
        val_losses.append(average_val_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {average_val_loss:.6f}')

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='blue')
    plt.title('Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss_MAE')
    plt.grid(True)
    plt.legend()
    plt.savefig('validation_loss_curve_MAE_dynamic_input_force.png')
    plt.show()

    torch.save(net.state_dict(), 'minkunet_model_MAE_dynamic_input_force.pth')

