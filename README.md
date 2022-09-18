# PointNet

This repository is to implement PointNet using [PyTorch](https://pytorch.org/) DL library, which is a deep learning network architecture proposed in 2016 by Stanford researchers and is the first neural network to handle directly 3D point clouds.

## Author

[Alex Choi](mailto:cinema4dr12@gmail.com)

## References
* [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593.pdf)
* [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf)
* [Medium Article on PointNet](https://medium.com/geekculture/understanding-3d-deep-learning-with-pointnet-fe5e95db4d2d)

## How to Run

### Clone from the repository

```
git clone https://git.jetbrains.space/alexchoi/personal-project/PointNet.git
```

### Install Python packages

```
cd PointNet
pip -install -r requirements.txt
```

### Dataset

You can download the ModelNet dataset from [here](https://modelnet.cs.princeton.edu#).

This code has been tested only for [ModelNet-10 dataset](http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip).

Or you can download all-in-one dataset from [this link](https://drive.google.com/file/d/1yTShG-2s3SIEW7C2MTCU32vHY5s2iL9w/view?usp=sharing).

Once you've finished downloading the dataset, please set the folder structure as follows.

* PointNet root path
  * Data
    * ModelNet10
    * ModelNet40

Of course, you can choose either one dataset of them.

### Configurations

This repository provide four main Python scripts:
* `trainer.py` which performs PointNet training.
* `batch-data-tester.py` which performs testing for the trained PointNet model from the `test` folder.
* `single-data-tester.py` which performs testing for a single point cloud data.
* `sample-data-visualizer.py` which performs the 3D visualizations of the sample point cloud data.

All configurations are defined in `Config/configs.yaml`.

## Results

After training has been done, you should be able to see the resultant folders:

* `runs`: Created by Tensorboard and can open it using the command line, `$ tensorboard --logdir runs`
* `Outputs`: Folders created followed by datetime of training started, which have `Figures` and `Models`.
  * `Figures`: Figures of confusion matrix as results of testing the trained network.
  * `Models`: Trained PointNet model files which has `best-model.pth` and `epoch-{####}.pth`.

## License

My code is released under MIT License (see [LICENSE](./LICENSE) file for details).