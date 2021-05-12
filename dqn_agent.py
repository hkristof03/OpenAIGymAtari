import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer
from preprocessing import stack_frame


class DQNAgent:
    def __init__(
        self, input_shape: tuple, action_size: int, seed: int, device: str,
        buffer_size: int, batch_size: int, gamma: float, lr: float, tau: float,
        update_every: int, replay_after: int, model: nn.Module,
        loss: str, **kwargs
    ):
        """Initialize an Agent object.

        Params
        ======
            input_shape (tuple): dimension of each state (C, H, W)
            action_size (int): dimension of each action
            seed (int): random seed
            device(string): Use Gpu or CPU
            buffer_size (int): replay buffer size
            batch_size (int):  minibatch size
            gamma (float): discount factor
            lr (float): learning rate
            update_every (int): how often to update the network
            replay_after (int): After which replay to be started
            model(Model): Pytorch Model
        """
        self.input_shape = input_shape
        self.action_size = action_size
        random.seed(seed)
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_every = update_every
        self.replay_after = replay_after
        self.DQN = model
        self.tau = tau

        # Q-Network
        self.policy_net = self.DQN(input_shape, action_size).to(self.device)
        self.target_net = self.DQN(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, seed,
                                   self.device)

        self.t_step = 0

        self.loss = loss
        self.criterion = nn.SmoothL1Loss() if loss == 'Huber' else nn.MSELoss()

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset
            # and learn
            if len(self.memory) > self.replay_after:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from policy model
        q_expected_current = self.policy_net(states)
        q_expected = q_expected_current.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.target_net(next_states).detach().max(1)[0]

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Compute loss
        loss = self.criterion(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.policy_net, self.target_net, self.tau)

    # θ'=θ×τ+θ'×(1−τ)
    def soft_update(self, policy_model, target_model, tau):
        for target_param, policy_param in zip(target_model.parameters(),
                                              policy_model.parameters()):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data)

    def evaluate_on_fixed_set(
        self,
        fixed_states: list
    ) -> float:
        """

        :param fixed_states: preprocessed fixed set of states
        :return:
        """
        action_values = []

        self.policy_net.eval()
        with torch.no_grad():

            state = stack_frame(None, fixed_states[0], True)

            for frame in fixed_states[1:]:
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(
                    self.device)
                max_action_value = np.max(
                    self.policy_net(state_tensor).cpu().data.numpy()
                )
                next_state = stack_frame(state, frame, False)
                state = next_state
                action_values.append(max_action_value)

        self.policy_net.train()

        return np.mean(action_values)
