import os
import yaml
import argparse

import torch
import pandas as pd

from dqn_agent import DQNAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pc', '--path_config', required=True, type=str,
                        help='path to the experiment config file')
    args = parser.parse_args()

    return args


def read_yaml(path_file: str) -> dict:
    """Reads the experiment's config file.

    :param path_file: Location of the yaml config file
    :return: Experiment config in a dictionary
    """
    path_dir = os.path.dirname(__file__)

    with open(os.path.join(path_dir, 'experiments', path_file)) as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    return config


def save_scores(conf: dict, stats: dict) -> None:

    df_stats = pd.DataFrame(stats)
    path_save = os.path.join(
        os.path.dirname(__file__),
        'artifacts',
        '_'.join(conf['tags']) + '.csv'
    )
    df_stats.to_csv(path_save, index=False)


def save_model(conf: dict, agent: DQNAgent) -> None:

    path_save_model = os.path.join(
        os.path.dirname(__file__),
        'artifacts',
        'saved_models',
        '_'.join(conf['tags'] + [conf['model_to_use'][1]]) + '.pth'
    )
    torch.save(agent.policy_net.state_dict(), path_save_model)
