import os
import torch
import random
import numpy as np
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from libs.models import network
from libs.utils import data_loader

def draw_confusion_matrix(cm, class_names, output_path="confusion_matrix.svg"):
    """Draw and save the confusion matrix as an SVG file with values as percentages."""
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8, 6))

    # Format annotations with the percentage sign
    annotations = np.array(["{:.2f}%".format(value) for row in cm_percentage for value in row])
    annotations = annotations.reshape(cm_percentage.shape)

    ax = sns.heatmap(cm_percentage, annot=annotations, fmt='', cmap='Blues', 
                     xticklabels=class_names, yticklabels=class_names,
                     annot_kws={"fontsize": 25},  # Increase font size of annotations
                     cbar_kws={'format': '%.0f%%'})  # Use integer percentage label

    # Adjust font sizes of labels
    plt.xlabel('Predicted Labels', fontsize=20)
    plt.ylabel('True Labels', fontsize=20)
    plt.xticks(fontsize=18)  # Increase font size of x-axis labels (class names)
    plt.yticks(fontsize=18)  # Increase font size of y-axis labels (class names)
    
    # Accessing the colorbar directly to adjust the label font size
    cbar = ax.collections[0].colorbar
    cbar.set_label('Percentage', fontsize=20)  # Set color bar label with desired font size

    plt.tight_layout()
    plt.savefig(output_path, format='svg')
    plt.close()


def test_net(params):
    device = torch.device("cuda:0" if torch.cuda.is_available() and params['use_gpu'] else "cpu")
    
    # Load the trained model
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

    model_path = params['test_model_path']
    assert os.path.exists(model_path), f"Model file not found: {model_path}"

    slip_detection_model.load_state_dict(torch.load(model_path, map_location=device))
    slip_detection_model.to(device)
    slip_detection_model.eval()

    # Dataloader for test set
    test_dataset = data_loader.Tactile_dataset(data_path=params['test_data_dir'])
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=params['num_workers'])

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        correct = 0
        total = 0
        for tactile_imgs, labels in test_data_loader:  # Unpack 3 values
            tactile_imgs = torch.stack(tactile_imgs)
            outputs = slip_detection_model(tactile_imgs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            if params['use_gpu']:
                labels = labels.cuda()
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
                
        print(f'Test Accuracy of the model on the {total} test images: {100 * correct / total} %')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    class_names = ["Slip", "Shear"]  # Modify based on your class labels
    #draw_confusion_matrix(cm, class_names, output_path="confusion_matrix_sim2sim.svg")

if __name__ == '__main__':
    params = {}
    params['rnn_input_size'] = 512
    params['rnn_hidden_size'] = 512
    params['num_classes'] = 2
    params['num_layers'] = 1
    params['use_gpu'] = torch.cuda.is_available()
    params['batch_size'] = 1
    params['num_workers'] = 1
    params['cnn'] = 'vgg_19' 
    params['pretrained'] = False
    params['dropout'] = 0
    params['test_data_dir'] = 'sim_data/test'
    #params['test_data_dir'] = 'real_data/test'
    params['test_model_path'] = 'model/Slip_detection_network_final.pth'  

    test_net(params)
