import os
import csv
import torch
from torch import optim, nn
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
from libs.models import network
from libs.utils import data_loader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_net(params):
    set_seed(42)
    slip_detection_model = network.Slip_detection_network_single_input(
        base_network=params['cnn'], 
        pretrained=params['pretrained'],
        rnn_input_size=params['rnn_input_size'],
        rnn_hidden_size=params['rnn_hidden_size'],
        rnn_num_layers=params['num_layers'],
        num_classes=params['num_classes'],
        use_gpu=params['use_gpu'],
        dropout=params['dropout']
    )
    if params['use_gpu']:
        slip_detection_model = slip_detection_model.cuda()

    if 'net_params' in params.keys():
        assert params['net_params'].endswith('.pth'), f"Wrong model path {params['net_params']}"
        net_params_state_dict = torch.load(params['net_params'])
        slip_detection_model.load_state_dict(net_params_state_dict)

    optimizer = optim.Adam(slip_detection_model.rnn_network.parameters(), lr=params['lr'])
    loss_function = nn.CrossEntropyLoss()

    full_dataset = data_loader.Tactile_dataset(data_path=params['train_data_dir'])
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_data_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                                   num_workers=params['num_workers'])
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=params['num_workers'])

    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    csv_path = os.path.join(params['save_dir'], 'training_log.csv')
    os.makedirs(params['save_dir'], exist_ok=True)

    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'])

        for epoch in range(params['epochs']):
            slip_detection_model.train()
            total_loss, total_acc, total = 0.0, 0.0, 0.0

            for data in train_data_loader:
                tactile_imgs, label = data
                if params['use_gpu']:
                    tactile_imgs = [img.cuda() for img in tactile_imgs]
                    label = label.cuda()
                output = slip_detection_model(tactile_imgs)
                loss = loss_function(output, label)
                slip_detection_model.zero_grad()
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(output.data, 1)
                total_acc += (predicted == label).sum().item()
                total_loss += loss.item()
                total += len(label)
            train_loss.append(total_loss / total)
            train_acc.append(total_acc / total)

            slip_detection_model.eval()
            val_total_loss, val_total_acc, val_total = 0.0, 0.0, 0.0
            with torch.no_grad():
                for data in val_data_loader:
                    tactile_imgs, label = data
                    if params['use_gpu']:
                        tactile_imgs = [img.cuda() for img in tactile_imgs]
                        label = label.cuda()
                    output = slip_detection_model(tactile_imgs)
                    loss = loss_function(output, label)
                    _, predicted = torch.max(output.data, 1)
                    val_total_acc += (predicted == label).sum().item()
                    val_total_loss += loss.item()
                    val_total += len(label)
            val_loss.append(val_total_loss / val_total)
            val_acc.append(val_total_acc / val_total)

            csv_writer.writerow([epoch + 1, train_loss[-1], train_acc[-1], val_loss[-1], val_acc[-1]])

            if epoch % params['print_interval'] == 0:
                print(f"[Epoch: {epoch+1}/{params['epochs']}] "
                      f"Train Loss: {train_loss[-1]:.3f}, Train Acc: {train_acc[-1]:.3f}, "
                      f"Val Loss: {val_loss[-1]:.3f}, Val Acc: {val_acc[-1]:.3f}")
                
    # Save trained model
    if 'save_dir' in params:
        model_path = os.path.join(params['save_dir'], f'Slip_detection_network.pth')
        torch.save(slip_detection_model.state_dict(), model_path)

    plt.figure(figsize=(10, 5))
    epochs_range = range(1, params['epochs'] + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(epochs_range, train_loss, 'r--', label='Train Loss')
    ax1.plot(epochs_range, val_loss, 'b--', label='Validation Loss')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    ax2.plot(epochs_range, train_acc, 'r-', label='Train Accuracy')
    ax2.plot(epochs_range, val_acc, 'b-', label='Validation Accuracy')
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper right')

    plt.title('Training & Validation Metrics')
    plt.grid(True)
    plt.savefig(os.path.join(params['save_dir'], 'training_curves.png'))
    plt.show()

if __name__ == '__main__':
    params = {
        'rnn_input_size': 512,
        'rnn_hidden_size': 512,
        'num_classes': 2,
        'num_layers': 1,
        'use_gpu': torch.cuda.is_available(),
        'epochs': 15,
        'print_interval': 1,
        'batch_size': 4,
        'num_workers': 1,
        'lr': 1e-3,
        'dropout': 0,
        'train_data_dir': 'sim_data/train',
        'cnn': 'vgg_19',
        'pretrained': True,
        'save_dir': 'model'
    }
    train_net(params)
