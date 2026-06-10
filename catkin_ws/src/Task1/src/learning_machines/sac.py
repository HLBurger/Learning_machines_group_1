import numpy as np
import random
import pickle
from pathlib import Path
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

from .constants_sac import (
    IR_indices,
    FRONT_INDICES,
    BACK_INDICES,
    LEFT_INDICES,
    RIGHT_INDICES,

    STATE_DIM,
    IR_MAX_VALUE,

    ACTION_DIM,
    MIN_WHEEL_SPEED,
    MAX_WHEEL_SPEED,
    ACTION_DURATION_MS,

    GAMMA,
    TAU,
    ACTOR_LR,
    CRITIC_LR,
    ALPHA_LR,
    BATCH_SIZE,
    REPLAY_BUFFER_SIZE,
    LEARNING_STARTS,
    UPDATES_PER_STEP,

    AUTO_ENTROPY_TUNING,
    TARGET_ENTROPY,
    INIT_ALPHA,

    HIDDEN_DIM,
    LOG_STD_MIN,
    LOG_STD_MAX,
)


RESULTS_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):

    """
    Gaussian policy for continuous SAC.
    Input:
        state: normalized IR sensor values, shape [batch, STATE_DIM]

    Output:
        sampled action in [-1, 1], log probability of action
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparameterization trick
        z = normal.rsample()

        # Squash action to [-1, 1]
        action = torch.tanh(z)

        # Log probability correction for tanh squashing
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def deterministic(self, state):
        mean, _ = self.forward(state)

        return torch.tanh(mean)

class Critic(nn.Module):
    """
    Q-network.

    Input:
        state, action

    Output:
        Q-value Q(s, a)
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.q_out(x)

class SAC_RL:
    def __init__(self, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.actor = Actor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=HIDDEN_DIM,
        ).to(self.device)

        self.critic1 = Critic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=HIDDEN_DIM,
        ).to(self.device)

        self.critic2 = Critic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=HIDDEN_DIM,
        ).to(self.device)

        self.target_critic1 = Critic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=HIDDEN_DIM,
        ).to(self.device)

        self.target_critic2 = Critic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=HIDDEN_DIM,
        ).to(self.device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=CRITIC_LR)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=CRITIC_LR)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.total_steps = 0

        if AUTO_ENTROPY_TUNING:
            self.log_alpha = torch.tensor(
                np.log(INIT_ALPHA),
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            )
            self.alpha_optimizer = Adam([self.log_alpha], lr=ALPHA_LR)

        else:
            self.alpha = INIT_ALPHA

    def preprocess_state(self, ir_values):

        """
        Convert raw IR values to normalized continuous state.

        Expected input:
            ir_values: list, tuple, or np.array of 8 raw IR readings

        Output:
            normalized np.array with values approximately in [0, 1]
        """

        state = np.asarray(ir_values, dtype=np.float32)

        if state.shape[0] != self.state_dim:
            raise ValueError(
                f"Expected {self.state_dim} IR values, got {state.shape[0]}"
            )

        state = np.clip(state, 0, IR_MAX_VALUE)
        state = state / IR_MAX_VALUE

        return state

    def scale_action_to_motor_speeds(self, action):

        """
        Convert SAC action from [-1, 1] to motor speeds.

        Example:
            -1 -> MIN_WHEEL_SPEED
             0 -> 0
            +1 -> MAX_WHEEL_SPEED
        """

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        motor_speeds = MIN_WHEEL_SPEED + (
            (action + 1.0) * 0.5 * (MAX_WHEEL_SPEED - MIN_WHEEL_SPEED)
        )

        left_speed = float(motor_speeds[0])
        right_speed = float(motor_speeds[1])

        return left_speed, right_speed, ACTION_DURATION_MS

    def select_action(self, ir_values, evaluate=False):

        """
        Select continuous wheel-speed action.

        During training:
            evaluate=False gives stochastic actions.

        During testing:
            evaluate=True gives deterministic actions.
        """

        state = self.preprocess_state(ir_values)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                action_tensor = self.actor.deterministic(state_tensor)
            else:
                action_tensor, _ = self.actor.sample(state_tensor)

        action = action_tensor.cpu().numpy()[0]

        return self.scale_action_to_motor_speeds(action), action

    def store_transition(self, state, action, reward, next_state, done):

        """
        Store normalized transition in replay buffer.

        Important:
            action should be the normalized SAC action in [-1, 1],
            not the scaled motor speeds.
        """

        state = self.preprocess_state(state)

        next_state = self.preprocess_state(next_state)

        self.replay_buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        self.total_steps += 1

    def current_alpha(self):
        if AUTO_ENTROPY_TUNING:
            return self.log_alpha.exp()

        return torch.tensor(self.alpha, dtype=torch.float32, device=self.device)

    def update(self):

        """
        Perform one SAC gradient update.

        Returns:
            Dictionary of losses, or None if replay buffer is too small.
        """

        if len(self.replay_buffer) < max(BATCH_SIZE, LEARNING_STARTS):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            BATCH_SIZE

        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        alpha = self.current_alpha()

        # -----------------------------

        # Critic update

        # -----------------------------

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q_min = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - dones) * GAMMA * (
                target_q_min - alpha * next_log_probs
            )

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # -----------------------------

        # Actor update

        # -----------------------------

        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new_min = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_probs - q_new_min).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # -----------------------------

        # Alpha / entropy update

        # -----------------------------

        alpha_loss_value = 0.0

        if AUTO_ENTROPY_TUNING:
            alpha_loss = -(
                self.log_alpha * (log_probs + TARGET_ENTROPY).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.item())

        # -----------------------------

        # Target critic soft update

        # -----------------------------

        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

        return {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.current_alpha().item()),
            "alpha_loss": alpha_loss_value,
        }

    def soft_update(self, source_network, target_network):

        for source_param, target_param in zip(
            source_network.parameters(),
            target_network.parameters(),
        ):

            target_param.data.copy_(
                TAU * source_param.data + (1.0 - TAU) * target_param.data
            )

    def save(self, filename="sac_agent.pt"):

        path = RESULTS_DIR / filename

        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "total_steps": self.total_steps,
        }

        if AUTO_ENTROPY_TUNING:
            checkpoint["log_alpha"] = self.log_alpha.detach().cpu()
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        torch.save(checkpoint, path)

        return path

    def load(self, filename="sac_agent.pt"):
        path = RESULTS_DIR / filename

        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)

        if AUTO_ENTROPY_TUNING and "log_alpha" in checkpoint:
            self.log_alpha = checkpoint["log_alpha"].to(self.device)
            self.log_alpha.requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=ALPHA_LR)

            if "alpha_optimizer" in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

        return path