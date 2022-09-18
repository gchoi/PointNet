import os
import yaml
import torch

def get_configurations(
    config_yaml_path: str,
    datetime_now: str
) -> dict:

    """ Gets configurations

    Params
    --------
        config_yaml_path (str): YAML configuration path
        datetime_now (str): Datetime now

    Returns
    --------
        configs (dict): Configuration info. read from YAML

    """

    assert(os.path.exists(config_yaml_path))

    with open(config_yaml_path) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    if os.name == 'nt':
        configs['data_pipeline']['num_workers'] = 0

    if configs['train']['seed']:
        torch.manual_seed(configs['train']['seed'])

    # check the validity
    assert(os.path.exists(configs['data']['data_path']))
    assert(os.path.exists(os.path.join(
        configs['data']['data_path'],
        configs['data']['sample_data_viz']
    )))

    # output model path
    output_path = os.path.join(
        configs['outputs']['root_path'],
        datetime_now,
        configs['outputs']['checkpoint_file_path']
    )
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    configs['output_model_path'] = output_path

    # output figure path
    output_path = os.path.join(
        configs['outputs']['root_path'],
        datetime_now,
        configs['outputs']['figure_path']
    )
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    configs['output_fig_path'] = output_path

    if configs['train']['from'] == 'checkpoint':
        assert(os.path.exists(configs['train']['pretrained_model']))

    return configs


def get_device(compute_device: str) -> str:

    """ Gets the computing device

    Params
    --------
        compute_device (str): Device name

    Returns
    --------
        device (str): Compute device (cpu/cuda/mps)

    """

    if compute_device == "cuda" and torch.cuda.is_available():
        return "cuda"

    if compute_device == "mps" and torch.backends.mps.is_available():
        return "mps"

    return "cpu"
