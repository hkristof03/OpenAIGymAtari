import math
from typing import Union

import gym
import torch
import numpy as np
from collections import deque

from models import dqn_cnn
from dqn_agent import DQNAgent
from preprocessing import preprocess_frame, stack_frame
from utils import read_yaml, parse_args, save_scores


def stack_frames(
    frames: Union[np.ndarray, None],
    state: np.ndarray,
    exclude: tuple,
    is_new: bool = False
):

    frame = preprocess_frame(state, exclude)
    frames = stack_frame(frames, frame, is_new)

    return frames


def decay_epsilon(conf: dict, current_episode: int) -> float:

    eps_start = conf['eps_start']
    eps_end = conf['eps_end']
    eps_decay = conf['eps_decay']

    current_epsilon = eps_end + (eps_start - eps_end) * math.exp(
        -1. * current_episode / eps_decay)

    return current_epsilon


def train(conf: dict) -> dict:

    env = gym.make(**conf['env'])
    env.seed(conf['seed'])

    conf['action_size'] = env.action_space.n
    conf['device'] = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    module, model_to_use = conf['model_to_use']
    model = getattr(globals()[module], model_to_use)
    conf['model'] = model
    crop_params = conf['preprocess']['exclude']
    n_episodes = conf['n_episodes']
    scores = []
    epsilons = []
    scores_window = deque(maxlen=20)
    eps = conf['eps_start']

    agent = DQNAgent(**exp_conf)

    for i_episode in range(1, n_episodes + 1):

        state = stack_frames(None, env.reset(), crop_params, True)
        score = 0
        epsilons.append(eps)
        eps = decay_epsilon(conf, i_episode)

        while True:
            env.render()
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            score += reward
            next_state = stack_frames(state, next_state, crop_params, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score

        print(f'Episode {i_episode}\tAverage Score: {np.mean(scores_window)}')

    env.close()

    return {
        'scores': scores,
        'epsilons': epsilons
    }


if __name__ == '__main__':

    arguments = parse_args()
    pc = arguments.path_config
    exp_conf = read_yaml(pc)

    stats = train(exp_conf)
    save_scores(exp_conf, stats)

    print(stats)

