"""
@Description: Module for testing
@Developed by: Alex Choi
@Date: 07/20/2022
@Contact: cinema4dr12@gmail.com
"""

# %% Import packages
import torch
from sklearn.metrics import confusion_matrix
from .network import PointNet
from .data import get_single_data
import numpy as np

def batch_test(
    network: PointNet,
    data_loader: torch.utils.data.DataLoader,
    device: str = 'cpu',
) -> np.ndarray:

    """ Performs test for the trained network

        Params
        --------
            network (PointNet): PointNet network
            data_loader (torch.utils.data.DataLoader): DataLoader for test dataset

        Returns
        --------
            cm (numpy.ndarray): Confusion matrix

    """

    network.eval()
    all_preds = []
    all_labels = []
    correct_sum = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if device == 'mps':
                # currently, 'mps' does not support torch.float64
                data['pointcloud'] = data['pointcloud'].to(torch.float32).to(device)
                data['category'] = data['category'].to(torch.int32).to(device)
                inputs = data['pointcloud'].to(device)
                labels = data['category'].to(device)
            else:
                inputs = data['pointcloud'].float().to(device)
                labels = data['category'].to(device)

            outputs, __, __ = network(inputs.transpose(1, 2))
            _, preds = torch.max(outputs.data, 1)
            all_preds += list(preds.cpu().numpy())
            all_labels += list(labels.cpu().numpy())
            correct_sum += (preds.cpu().numpy() == labels.cpu().numpy()).sum()
            test_acc = correct_sum / len(all_preds) * 100.0
            print('Batch [%4d / %4d]\t Test accuracy up to current batch: %5.1f %%'\
                  % (i + 1, len(data_loader), test_acc))

    cm = confusion_matrix(all_labels, all_preds)
    return cm


def single_data_test(
    network: PointNet,
    classes: dict,
    data_path: str,
    device: str
) -> str:

    """ Predicts for the single point cloud data

        Params
        --------
            network (PointNet): PointNet network
            classes (dict): Known classes of point cloud data
            data_path (str): Path to the point cloud data file
            device (str): Computing device (cpu/cuda/mps)

        Returns
        --------
            category (str): Prediction result as category

    """

    _data = get_single_data(data_path)
    network.eval()
    with torch.no_grad():
        if device == 'mps':
            # currently, 'mps' does not support torch.float64
            _data = _data.to(torch.float32).to(device)
            _input = _data.to(device)
        else:
            _input = _data.float().to(device)

        _input = torch.unsqueeze(_input, 0)
        _output, __, __ = network(_input.transpose(1, 2))
        _, pred = torch.max(_output.data, 1)
        pred = pred.cpu().numpy()[0]

        for category in classes.keys():
            if classes[category] == pred:
                break

    return category