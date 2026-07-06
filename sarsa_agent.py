"""
Deep SARSA Agent
================================================================

【SARSAとDQNの違い】

  Q-learning (DQN) : オフポリシー (Off-policy)
    target = r + γ * max_a Q(s', a)     ← 次状態の最大Q値を使う
    → 実際にどの行動を取るかに関わらず、「最良の行動」を仮定して更新

  SARSA            : オンポリシー (On-policy)
    target = r + γ * Q(s', a')          ← 実際に選んだ次行動のQ値を使う
    a' は現在の epsilon-greedy 方策で選択
    → 実際に取る行動（探索を含む）をそのまま反映して更新

【特性の違い】
  DQN   : 楽観的・攻撃的。最大Q値を常に目指すため高スコアを狙える
           ただしQ値過大評価が起きやすい
  SARSA : 保守的・安全。実際の行動方策を反映するためリスク回避的
           探索（ランダム行動）の結果も学習に組み込まれる

  → パックマンでは「ゴーストを避けながら慎重にエサを取る」
    SARSAの方が安定した行動が出やすい傾向がある
================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class SARSANetwork(nn.Module):
    """DQNと同じネットワーク構造（比較のため同一にする）"""
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


class SARSAReplayBuffer:
    """SARSA用リプレイバッファ: (s, a, r, s', a', done) を保存"""
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, next_action, done):
        self.buffer.append((state, action, reward, next_state, next_action, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, next_actions, dones = zip(*batch)
        return (
            np.array(states,       dtype=np.float32),
            np.array(actions,      dtype=np.int64),
            np.array(rewards,      dtype=np.float32),
            np.array(next_states,  dtype=np.float32),
            np.array(next_actions, dtype=np.int64),
            np.array(dones,        dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class SARSAAgent:
    """
    Deep SARSA エージェント

    DQNとの主な実装上の違い:
      1. リプレイバッファに next_action も保存
      2. 学習時のターゲット計算で max ではなく Q(s', a') を使用
      3. Target Network なし（オンポリシーのため不要）
         ※ 安定化のため Semi-gradient SARSA として実装
    """

    def __init__(self, obs_size, n_actions, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 batch_size=64):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.steps         = 0

        self.device = self._get_device()
        print(f"[SARSA] Using device: {self.device}")

        self.net       = SARSANetwork(obs_size, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.buffer    = SARSAReplayBuffer()
        self.loss_history = []

    @staticmethod
    def _get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            import torch_directml
            dml = torch_directml.device()
            torch.tensor([1.0]).to(dml)
            return dml
        except Exception:
            pass
        return torch.device("cpu")

    def get_qvalues(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.net(state_t)
        return q.squeeze(0).cpu().numpy()

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        q_values = self.get_qvalues(state)
        return int(np.argmax(q_values))

    def push(self, state, action, reward, next_state, next_action, done):
        """SARSAは (s, a, r, s', a') のタプルを保存"""
        self.buffer.push(state, action, reward, next_state, next_action, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, next_actions, dones = \
            self.buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        next_actions_t= torch.LongTensor(next_actions).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # 現在のQ値 Q(s, a)
        q_values = self.net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # SARSAターゲット: r + γ * Q(s', a')  ← 実際に選んだ次行動のQ値
        with torch.no_grad():
            next_q = self.net(next_states_t).gather(1, next_actions_t.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.SmoothL1Loss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss.item()

    def save(self, path):
        torch.save({
            "net"      : self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon"  : self.epsilon,
            "steps"    : self.steps,
        }, path)
        print(f"[SARSA] Model saved: {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(ckpt["net"])
        self.net.to(self.device)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.steps   = ckpt["steps"]
        print(f"[SARSA] Model loaded: {path}  (epsilon={self.epsilon:.3f})")
