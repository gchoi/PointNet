"""
@Description: Module for the network model
@Developed by: Alex Choi
@Date: 07/20/2022
@Contact: cinema4dr12@gmail.com
"""

# %% Import packages
import os
import torch
from .network import PointNet

def pointnet_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    m3x3: torch.Tensor,
    m64x64: torch.Tensor,
    alpha: float = 0.0001,
    device: str = 'cpu'
):

    """ Calculates the pointnet loss

        Params
        --------
            outputs (torch.Tensor): Output tensor for the current batch
            labels (torch.Tensor): Labels for the current batch
            m3x3 (torch.Tensor): M3x3 matrix for the current batch
            m64x64 (torch.Tensor): M64x64 matrix for the current batch
            alpha (float): Learning rate
            device (str): Computing device (cpu/cuda/mps

        Returns
        --------
            loss (float): Pointnet loss

    """

    criterion = torch.nn.NLLLoss()
    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)

    id3x3 = id3x3.to(device)
    id64x64 = id64x64.to(device)

    diff3x3 = id3x3-torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64-torch.bmm(m64x64, m64x64.transpose(1, 2))
    loss = criterion(outputs, labels) + alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)
    return loss


def pointnet_model(
    num_classes: int,
    device: str = 'cpu',
    mode: str = 'train',
    train_from: str = 'checkpoint',
    model: str = ''
) -> tuple:

    """ Loads the PointNet network with optimizer and number of epochs trained

        Params
        --------
            num_classes (int): Number of classes
            device (str): Computing device (cpu/cuda/mps)
            mode (str): Options either of train/valid/test
            train_from (str): Train mode starting from scratch or an existing checkpoint file
            model (str): Path to the model (pre-trained or trained)

        Returns
        --------
            tuple:
                pointnet (PointNet): PointNet network object
                optimizer (torch.optim.adam.Adam): Adam optimizer
                epochs_trained (int):
                    0: Not yet trained (training will start from 0 epoch)
                    > 0: Number of epochs trained

    """

    # Load a pre-trained model if it exists
    if not mode == 'train' and train_from == 'scratch':
        assert (os.path.exists(model))

    pointnet = PointNet(num_classes=num_classes, device=device)
    epochs_trained = 0
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.00025)

    if mode == 'train':
        if train_from == 'checkpoint':
            loaded_model = torch.load(model, map_location=device)
            # epochs trained
            if 'epoch' in loaded_model.keys():
                epochs_trained = loaded_model['epoch']
            # network state
            if 'network_state_dict' in loaded_model.keys():
                pointnet.load_state_dict(loaded_model['network_state_dict'])
            else:
                pointnet.load_state_dict(loaded_model)
            # optimizer state
            if 'optimizer_state_dict' in loaded_model.keys():
                optimizer.load_state_dict(loaded_model['optimizer_state_dict'])

        elif train_from == 'scratch':
            pass

        else:
            raise ValueError

    elif mode == 'test':
        loaded_model = torch.load(model, map_location=device)
        # epochs trained
        if 'epoch' in loaded_model.keys():
            epochs_trained = loaded_model['epoch']
        # network state
        if 'network_state_dict' in loaded_model.keys():
            pointnet.load_state_dict(loaded_model['network_state_dict'])
        else:
            pointnet.load_state_dict(loaded_model)
    else:
        raise ValueError

    pointnet.to(device)

    return pointnet, optimizer, epochs_trained


def save_model(
    epoch: int,
    network: PointNet,
    optimizer: torch.optim.Adam,
    loss: float,
    best_acc: float,
    is_save_best_model: bool = False,
    checkpoint_file_path: str = None,
    checkpoint_file_ext: str = None
):

    """ Saves the trained model

        Params
        --------
            epoch (int): Current epoch
            network (PointNet): PointNet network
            optimizer (torch.optim.adam.Adam): Optimizer
            loss (float): Average loss
            best_acc (float): Best accuracy
            is_save_best_model (bool): Option to save the best model
            checkpoint_file_path (str): Path to the root of the model file
            checkpoint_file_ext (str): File extension of the model file

    """

    if is_save_best_model:
        torch.save(
            {
                'epoch': epoch,
                'network_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'best_acc': best_acc
            },
            os.path.join(
                checkpoint_file_path,
                f'best-model.{checkpoint_file_ext}'
            )
        )
    else:
        torch.save(
            {
                'epoch': epoch,
                'network_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'best_acc': best_acc
            },
            os.path.join(
                checkpoint_file_path,
                'epoch-{:05d}.{}'.format(epoch, checkpoint_file_ext)
            )
        )

    return