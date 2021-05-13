import math
from typing import Union

import gym
from gym.wrappers.time_limit import TimeLimit
import torch
import numpy as np
from collections import deque

from models import dqn_cnn
from dqn_agent import DQNAgent
from preprocessing import preprocess_frame, stack_frames
from utils import read_yaml, parse_args, save_scores, save_model


def collect_fixed_set_of_states(conf: dict, env: TimeLimit) -> list:
    # Collect samples to evaluate the agent on a fixed set of samples
    # (DQN paper). Collect a fixed set of states by running a random policy
    # before training starts and track the average of the maximum predicted
    # Q for these states.
    env.reset()
    exclude = conf['preprocess']['exclude']
    fixed_states = []

    while True:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        preprocessed_state = preprocess_frame(state, exclude)
        fixed_states.append(preprocessed_state)
        if done:
            break
    env.close()
    print(f'Collected {len(fixed_states)} fixed set of states!')

    return fixed_states


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
    # Evaluate the agent based on the mean of the Q values on the fixed set
    # of states
    fixed_states = collect_fixed_set_of_states(conf, env)
    average_action_values = []

    agent = DQNAgent(**exp_conf)
    agent_hps = np.inf

    for i_episode in range(1, n_episodes + 1):

        state = stack_frames(None, env.reset(), crop_params, True)
        score = 0
        epsilons.append(eps)
        eps = decay_epsilon(conf, i_episode)

        while True:
            # env.render()
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)

            if reward == 0.0 and not done:
                reward += -0.01

            if agent_hps == np.inf:
                agent_hps = info['ale.lives']

            elif info['ale.lives'] < agent_hps:
                reward += -50.0
                agent_hps += -1

            score += reward
            next_state = stack_frames(state, next_state, crop_params, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        avg_av = agent.evaluate_on_fixed_set(fixed_states)
        average_action_values.append(avg_av)

        print(
            f'Episode {i_episode}\tAverage Score: '
            f'{round(np.mean(scores_window),4)}\tEpsilon: {round(eps, 4)}\t'
            f'Average Q value: {round(avg_av, 4)}'
        )

        if i_episode % conf['save_every'] == 0 and i_episode > 0:
            print(f'Saving model at iteration: {i_episode}')
            save_model(conf, agent)

    env.close()

    return {
        'scores': scores,
        'epsilons': epsilons,
        'avg_action_values': average_action_values
    }


if __name__ == '__main__':

    arguments = parse_args()
    pc = arguments.path_config
    exp_conf = read_yaml(pc)

    stats = train(exp_conf)
    save_scores(exp_conf, stats)


