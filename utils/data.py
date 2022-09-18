import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .torch_utils import default_transforms
from .utils import (
    PointSampler,
    Normalize,
    RandRotation_z,
    RandomNoise
)
from .torch_utils import ToTensor
from .visualize import read_off


class PointCloudData(Dataset):
    def __init__(
        self,
        root_dir: str,
        folder: str = "train",
        transform: transforms.Compose = default_transforms()
    ) -> None:

        """ Class initializer

            Params
            --------
                root_dir (str): Path to data root
                folder (str): Name of 'train' folder
                transform (torchvision.transforms.transforms.Compose): Data transform

        """

        self.root_dir = root_dir
        folders = [
            _dir for _dir in sorted(os.listdir(root_dir))\
            if os.path.isdir(os.path.join(root_dir, _dir))
        ]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.files = []

        for category in self.classes.keys():
            new_dir = os.path.join(root_dir, category, folder)
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = dict()
                    sample['pcd_path'] = os.path.join(new_dir, file)
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self) -> int:

        """ Returns the length of files
        """

        return len(self.files)

    def __preproc__(self, file) -> torch.Tensor:

        """ Calculates the transformation of the pointcloud data

            Params
            --------
                file (_io.TextIOWrapper): File IO

            Returns
            --------
                pointcloud (torch.Tensor): Transformed point cloud data

        """

        verts, faces = read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx: int):

        """ Returns pointcloud data and category

            Params
            --------
                idx (int): Index of data

            Returns
            --------
                ret (dict): Pointcloud and its category

        """

        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)

        ret = {
            'pointcloud': pointcloud,
            'category': self.classes[category]
        }
        return ret


def get_dataset(
    data_path: str,
    folder: str,
    dataset_type: str = 'train'
) -> PointCloudData:

    """ Gets dataset

            Params
            --------
                data_path (str):
                folder (str):
                dataset_type (str):

            Returns
            --------
                ds (PointCloudData): PointCloudData class (dataset)

        """

    ds = None
    if dataset_type == 'train':
        train_transforms = transforms.Compose(
            [
                PointSampler(1024),
                Normalize(),
                RandRotation_z(),
                RandomNoise(),
                ToTensor()
            ]
        )
        ds = PointCloudData(
            root_dir=data_path,
            folder=folder,
            transform=train_transforms
        )
    elif dataset_type == 'valid':
        valid_transforms = transforms.Compose(
            [
                PointSampler(1024),
                Normalize(),
                ToTensor()
            ]
        )
        ds = PointCloudData(
            root_dir=data_path,
            folder=folder,
            transform=valid_transforms
        )
    elif dataset_type == 'test':
        test_transforms = transforms.Compose(
            [
                PointSampler(1024),
                Normalize(),
                ToTensor()
            ]
        )
        ds = PointCloudData(
            root_dir=data_path,
            folder=folder,
            transform=test_transforms
        )
    else:
        raise ValueError('dataset type mismatches.')

    inv_classes = {i: cat for cat, i in ds.classes.items()};
    print(inv_classes)

    print(f'\nDataset size for {dataset_type}: ', len(ds))
    print('Number of classes: ', len(ds.classes))

    return ds


def get_dataloader(
    data_path: str,
    folder: str,
    dataset_type: str = "train",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: bool = True
) -> DataLoader:

    """ Gets dataloader

        Params
        --------
            data_path (str): Path to data root
            folder (str): Name of 'train' folder
            dataset_type (str): Type of dataset (train/valid/test)
            batch_size (int): Batch size
            num_workers (int): Number of workers for data pipeline
            pin_memory (bool): Use pin memory?
            shuffle (bool): Shuffle data?

        Returns
        --------
            data_loader (torch.utils.data.dataloader.Dataloader): Dataloader

    """

    dataset = get_dataset(
        data_path=data_path,
        folder=folder,
        dataset_type=dataset_type
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return data_loader


def get_single_data(file: str) -> PointCloudData:

    """ Gets a single pointcloud data from the file path

        Params
        --------
            file (str): Path to the point cloud data

        Returns
        --------
            pointcloud (torch.Tensor): Transformed pointcloud data

    """

    test_transforms = transforms.Compose(
        [
            PointSampler(1024),
            Normalize(),
            ToTensor()
        ]
    )

    with open(file, 'r') as f:
        verts, faces = read_off(f)
        pointcloud = test_transforms((verts, faces))

    return pointcloud
