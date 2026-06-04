"""
DQN Agent for Pac-Man
Double DQN + Experience Replay + Target Network
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class DQNNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, obs_size, n_actions, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 batch_size=64, target_update_freq=500):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_net = DQNNetwork(obs_size, n_actions).to(self.device)
        self.target_net = DQNNetwork(obs_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.loss_history = []

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax(dim=1).item()

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states_t     = torch.FloatTensor(states).to(self.device)
        actions_t    = torch.LongTensor(actions).to(self.device)
        rewards_t    = torch.FloatTensor(rewards).to(self.device)
        next_states_t= torch.FloatTensor(next_states).to(self.device)
        dones_t      = torch.FloatTensor(dones).to(self.device)

        # Double DQN
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.SmoothL1Loss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss.item()

    def save(self, path):
        torch.save({
            "policy": self.policy_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
        }, path)
        print(f"Model saved: {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.steps = ckpt["steps"]
        print(f"Model loaded: {path}  (epsilon={self.epsilon:.3f})")
