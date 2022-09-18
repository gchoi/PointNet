"""
@Description: Main script for testing a single point cloud datum
@Developed by: Alex Choi
@Date: 07/20/2022
@Contact: cinema4dr12@gmail.com
"""

# %% Import packages
from logger import logging
from datetime import datetime
from utils import (
    get_classes,
    pointnet_model,
    get_configurations,
    get_device,
    single_data_test
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
        f"Selected Computing Device:\t       {configs['computing_device']}\n"
        "\n"

        f"TEST INFO:\n"
        f"\tTrained Model Path:            {configs['single_test']['trained_model_path']}\n"
        f"\tTest Data Path:                {configs['single_test']['test_data_path']}\n"
        "\n"
    )

    # %% what kinds of classes do we have?
    classes: dict = get_classes(configs=configs)
    logging.info(f"Classes: {classes}")

    # %% let's test the trained network
    logging.info("Testing started...")
    num_classes = len(classes)
    pointnet, optimizer, epochs_trained = pointnet_model(
        num_classes=num_classes,
        device=device,
        mode='test',
        model=configs['single_test']['trained_model_path']
    )
    pred = single_data_test(
        network=pointnet,
        classes=classes,
        data_path=configs['single_test']['test_data_path'],
        device=device
    )
    print(f"Prediction: {pred}")
    logging.info("Testing DONE!")


if __name__ == '__main__':
    main()
