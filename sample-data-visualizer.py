"""
@Description: Main script for sample data visualization
@Developed by: Alex Choi
@Date: 09/16/2022
@Contact: cinema4dr12@gmail.com
"""

# %% Import packages
from logger import logging
from datetime import datetime
from utils import (
    get_configurations,
    viz_sample_data
)


def main() -> None:
    datetime_now = datetime.now().strftime('%Y%m%d-%H%M%S')

    # %% set configurations
    YAML_CONFIG_PATH = "./Config/configs.yaml"
    configs = get_configurations(
        config_yaml_path=YAML_CONFIG_PATH,
        datetime_now=datetime_now
    )

    logging.info(
        f"\tSample Data:                   {configs['data']['sample_data_viz']}\n"
        "\n"
    )


    # %% visualize a sample data
    logging.info("Visualizing sample data...")
    viz_sample_data(configs=configs)


if __name__ == '__main__':
    main()
