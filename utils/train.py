"""
@Description: Module for training
@Developed by: Alex Choi
@Date: 07/20/2022
@Contact: cinema4dr12@gmail.com
"""

# %% Import packages
from logger import logging
import torch
from .model import save_model, pointnet_loss
from .network import PointNet
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

def train(
    epochs_trained: int,
    num_epochs: int,
    network: PointNet,
    optimizer: torch.optim.Adam,
    loss_func: pointnet_loss,
    checkpoint_file_path: str,
    checkpoint_file_ext: str,
    writer: SummaryWriter,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    device: str = 'cpu'
) -> None:

    """ Trains the PointNet network

        Params
        --------

            epochs_trained (int): Number of epochs trained from previous network model
            num_epochs (int): Number of epochs to train
            network (PointNet): PointNet network
            optimizer (torch.optim.Adam): Adam optimizer - custom optimizer will be supported in the future
            loss_func (pointnet_loss): PointNet loss
            checkpoint_file_path (str): Path to the root of the model file
            checkpoint_file_ext (str): File extension of the model file
            writer (torch.utils.tensorboard.writer.SummaryWriter): Tensorboard summary writer
            train_loader (torch.utils.data.dataloader.DataLoader): DataLoader for train
            val_loader (torch.utils.data.dataloader.DataLoader): DataLoader for validation
            device (str): Computing device (cpu/cuda/mps)

    """

    if epochs_trained > num_epochs:
        logging.error("epochs_trained must be less than or equal to num_epochs.")
    assert(epochs_trained <= num_epochs)

    best_acc = 0.0
    for epoch in range(epochs_trained, num_epochs):
        network.train()
        running_loss = 0.0
        avg_loss = 0.0
        logging.info(f"Training for epoch#{epoch + 1}")
        for i, data in enumerate(train_loader, 0):
            if device == 'mps':
                # currently, 'mps' does not support torch.float64
                data['pointcloud'] = data['pointcloud'].to(torch.float32).to(device)
                data['category'] = data['category'].to(torch.int32).to(device)
                inputs = data['pointcloud'].to(device)
                labels = data['category'].to(device)
            else:
                inputs = data['pointcloud'].to(device).float()
                labels = data['category'].to(device)

            optimizer.zero_grad()
            outputs, m3x3, m64x64 = network(inputs.transpose(1, 2))

            loss = loss_func(outputs, labels, m3x3, m64x64, alpha=0.0001, device=device)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)
            print('\t[Epoch: %d / %d, Batch: %4d / %4d], Average Loss for current epoch: %.3f' %
                (epoch + 1, num_epochs, i + 1, len(train_loader), avg_loss))

        # validation
        logging.info(f"Validation for epoch#{epoch + 1}")
        network.eval()
        correct = total = 0
        val_acc = 0.0
        if val_loader:
            with torch.no_grad():
                for i, data in enumerate(val_loader, 0):
                    if device == 'mps':
                        # currently, 'mps' does not support torch.float64
                        data['pointcloud'] = data['pointcloud'].to(torch.float32)
                        data['category'] = data['category'].to(torch.int32)
                        inputs = data['pointcloud'].to(device)
                        labels = data['category'].to(device)
                    else:
                        inputs = data['pointcloud'].to(device).float()
                        labels = data['category'].to(device)

                    outputs, __, __ = network(inputs.transpose(1, 2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    print('\t[Epoch: %d / %d, Batch: %4d / %4d], Correct: %d / %d ' %
                          (epoch + 1, num_epochs, i + 1, len(val_loader), correct, total))

            val_acc = 100. * correct / total
            print('Validation accuracy: ' + '{:5.3f} %'.format(val_acc))

            if val_acc > best_acc:
                best_acc = val_acc
                print('Best validation accuracy: ' + '{:5.3f} %'.format(best_acc))
                # save the best model
                save_model(
                    epoch=epoch,
                    network=network,
                    optimizer=optimizer,
                    loss=avg_loss,
                    best_acc=best_acc,
                    is_save_best_model=True,
                    checkpoint_file_path=checkpoint_file_path,
                    checkpoint_file_ext=checkpoint_file_ext
                )
                print("Best model saved!")

        # save the model
        save_model(
            epoch=epoch,
            network=network,
            optimizer=optimizer,
            loss=avg_loss,
            best_acc=best_acc,
            is_save_best_model=False,
            checkpoint_file_path=checkpoint_file_path,
            checkpoint_file_ext=checkpoint_file_ext
        )

        # write train information
        writer.add_scalar('train/loss', avg_loss, epoch + 1)
        writer.add_scalar('validation/accuracy', val_acc, epoch + 1)

    return