"""
@Description: Module for visualization
@Developed by: Alex Choi
@Date: 07/20/2022
@Contact: cinema4dr12@gmail.com
"""

# %% Import packages
import os
from typing import Tuple
import numpy as np
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
import itertools
from utils import (
    PointSampler,
    Normalize,
    RandRotation_z,
    RandomNoise
)

def read_off(file) -> Tuple[list, list]:

    """ Reads 'OFF' headers

        Params
        --------
            file (_io.TextIOWrapper): file IO

        Returns
        --------
            verts (list): List of vertices
            faces (list): List of faces

    """

    if 'OFF' != file.readline().strip():
        raise 'Not a valid OFF header'
    n_verts, n_faces, _ = tuple([
        int(s) for s in file.readline().strip().split(' ')
    ])
    verts = [
        [float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)
    ]
    faces = [
        [int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)
    ]
    return verts, faces


def visualize_rotate(data: list) -> go.Figure:

    """ Visualize

        Params
        --------
            data (list): Mesh data

        Returns
        --------
            fig (plotly.graph_objects.Figure): Plotly figure object

    """

    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames = []

    def rotate_z(x, y, z, theta):
        w = x + (1j * y)
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(
            dict(
                layout=dict(
                    scene=dict(
                        camera=dict(
                            eye=dict(
                                x=xe,
                                y=ye,
                                z=ze
                            )
                        )
                    )
                )
            )
        )

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    y=1,
                    x=0.8,
                    xanchor='left',
                    yanchor='bottom',
                    pad=dict(t=45, r=10),
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[
                                None,
                                dict(
                                    frame=dict(
                                        duration=50,
                                        redraw=True
                                    ),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    mode='immediate'
                                )
                            ]
                        )
                    ]
                )
            ]
        ),
        frames=frames
    )

    return fig


def pcshow(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray
) -> None:

    """ Shows the point cloud data

        Params
        --------
            xs (numpy.ndarray): x-coordinate values of the point cloud data
            ys (numpy.ndarray): y-coordinate values of the point cloud data
            zs (numpy.ndarray): z-coordinate values of the point cloud data

    """

    data = [go.Scatter3d(x=xs, y=ys, z=zs, mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(
        marker=dict(
            size=2,
            line=dict(
                width=2,
                color='DarkSlateGrey'
            )
        ),
        selector=dict(mode='markers')
    )
    fig.show()
    return


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list,
    output_fig_path: str,
    output_fig_ext: str,
    normalize: bool = False,
    title: str = 'Confusion matrix',
    cmap: matplotlib.colors.LinearSegmentedColormap = plt.cm.Blues
) -> None:

    """ Shows confusion matrix plot

        Params
        --------
            cm (numpy.ndarray): Confusion matrix
            classes (list): List of classes
            output_fig_path (str): Output path to save the confusion matrix figure
            output_fig_ext (str): File extension of the output confusion matrix figure
            normalize (bool): Option for normalization
            title (str): Title of the plot
            cmap (matplotlib.colors.LinearSegmentedColormap): Color map

    """

    plt.figure(figsize=(8, 8))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    save_path = os.path.join(
        output_fig_path,
        f"{title}.{output_fig_ext}"
    )
    plt.savefig(save_path)
    plt.show()
    return


def viz_sample_data(configs: dict) -> None:

    """ Visualizes sample data

        Params
        --------
            configs (dict): YAML configuration path

    """

    with open(
        os.path.join(
            configs['data']['data_path'],
            configs['data']['sample_data_viz']
        ), 'r'
    ) as f:
        verts, faces = read_off(f)

    i, j, k = np.array(faces).T
    x, y, z = np.array(verts).T

    visualize_rotate(
        [
            go.Mesh3d(
                x=x, y=y, z=z,
                color='yellowgreen',
                opacity=0.50,
                i=i, j=j, k=k
            )
        ]
    ).show()

    visualize_rotate(
        [
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers'
            )
        ]
    ).show()

    pcshow(x, y, z)

    pointcloud = PointSampler(3000)((verts, faces))
    pcshow(*pointcloud.T)

    norm_pointcloud = Normalize()(pointcloud)
    pcshow(*norm_pointcloud.T)

    rot_pointcloud = RandRotation_z()(norm_pointcloud)
    noisy_rot_pointcloud = RandomNoise()(rot_pointcloud)
    pcshow(*noisy_rot_pointcloud.T)
    return