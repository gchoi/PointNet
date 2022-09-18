"""
@Description: Module of Torch utilities
@Developed by: Alex Choi
@Date: 07/20/2022
@Contact: cinema4dr12@gmail.com
"""

# %% Import packages
import numpy as np
import torch
from torchvision import transforms
from utils import (
    PointSampler,
    Normalize
)


class ToTensor(object):
    def __call__(self, pointcloud: np.ndarray) -> torch.Tensor:

        """ Converts data type of pointcloud from numpy array to torch tensor

            Params
            --------
                pointcloud (numpy.ndarray): Point cloud data array

            Returns
            --------
                torch.Tensor

        """

        assert len(pointcloud.shape) == 2
        return torch.from_numpy(pointcloud)


def default_transforms() -> transforms.Compose:

    """ Returns data transforms

        Returns
        --------
            torchvision.transforms.transforms.Compose

    """

    return transforms.Compose(
        [
            PointSampler(1024),
            Normalize(),
            ToTensor()
        ]
    )