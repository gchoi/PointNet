"""
@Description: Main script for testing
@Developed by: Alex Choi
@Date: 07/20/2022
@Contact: cinema4dr12@gmail.com
"""

# %% Import packages
from logger import logging
from datetime import datetime
import matplotlib.pyplot as plt
from utils import (
    get_classes,
    get_dataloader,
    pointnet_model,
    batch_test,
    get_configurations,
    get_device,
    plot_confusion_matrix
)


def main() -> None:
    datetime_now = datetime.now().strftime('%Y%m%d-%H%M%S')

    # %% set configurations
    YAML_CONFIG_PATH = "./Config/configs.yaml"
    configs = get_configurations(
        config_yaml_path=YAML_CONFIG_PATH,
        datetime_now=datetime_now
    )
    device = get_device(compute_device=configs['computing_device'])

    logging.info(
        f"DATA INFO:\n"
        f"\tData Root Path:                {configs['data']['data_path']}\n"
        f"\tTest Directory:                {configs['data']['test_dir']}\n"
        "\n"

        f"Selected Computing Device:\t       {configs['computing_device']}\n"
        "\n"

        f"TEST INFO:\n"
        f"\tTrained Model Path:            {configs['batch_test']['trained_model_path']}\n"
        f"\tBatch Size:                    {configs['batch_test']['batch_size']}\n"
        "\n"
    )

    # %% what kinds of classes do we have?
    classes = get_classes(configs=configs)
    logging.info(f"Classes: {classes}")

    # %% data loaders
    logging.info("Loading the data...")
    test_loader = get_dataloader(
        data_path=configs['data']['data_path'],
        folder=configs['data']['test_dir'],
        dataset_type="test",
        batch_size=configs['batch_test']['batch_size'],
        num_workers=configs['data_pipeline']['num_workers'],
        pin_memory=configs['data_pipeline']['pin_memory'],
        shuffle=configs['batch_test']['shuffle']
    )


    # %% let's test the trained network
    logging.info("Testing started...")
    num_classes = len(classes)
    pointnet, optimizer, epochs_trained = pointnet_model(
        num_classes=num_classes,
        device=device,
        mode='test',
        model=configs['batch_test']['trained_model_path']
    )
    cm = batch_test(
        network=pointnet,
        data_loader=test_loader,
        device=device
    )
    logging.info("Testing DONE!")

    # %% plot the confusion matrix
    plot_confusion_matrix(
        cm=cm,
        classes=list(classes.keys()),
        output_fig_path=configs['output_fig_path'],
        output_fig_ext=configs['outputs']['figure_file_ext'],
        normalize=True,
        title='Normalized Confusion matrix',
        cmap=plt.cm.Blues
    )

    plot_confusion_matrix(
        cm=cm,
        classes=list(classes.keys()),
        output_fig_path=configs['output_fig_path'],
        output_fig_ext=configs['outputs']['figure_file_ext'],
        normalize=False,
        title='Unnormalized Confusion matrix',
        cmap=plt.cm.Blues
    )


if __name__ == '__main__':
    main()
