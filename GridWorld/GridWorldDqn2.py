import gym
import torch
import numpy as np
import random
import os
from typing import Tuple, List, NamedTuple, Dict
from math import floor
from torch.nn import Parameter
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.registry import default_registry
from mlagents_envs.environment import ActionTuple, BaseEnv

class VisualQNetwork(torch.nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], encoding_size: int, output_size: int):
        super(VisualQNetwork, self).__init__()
        height = input_shape[1]
        width = input_shape[2]
        initial_channels = input_shape[0]
        conv_1_hw = self.conv_output_shape((height, width), 8, 4)
        conv_2_hw = self.conv_output_shape(conv_1_hw, 4, 2)
        self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32
        self.conv1 = torch.nn.Conv2d(initial_channels, 16, [8, 8], [4, 4])
        self.conv2 = torch.nn.Conv2d(16, 32, [4, 4], [2, 2])
        self.dense1 = torch.nn.Linear(self.final_flat, encoding_size)
        self.dense2 = torch.nn.Linear(encoding_size, output_size)

    def forward(self, visual_obs: torch.tensor):
        conv_1 = torch.relu(self.conv1(visual_obs))
        conv_2 = torch.relu(self.conv2(conv_1))
        hidden = self.dense1(conv_2.reshape([-1, self.final_flat]))
        hidden = torch.relu(hidden)
        hidden = self.dense2(hidden)
        return hidden

    @staticmethod
    def conv_output_shape(h_w: Tuple[int, int], kernel_size: int = 1, stride: int = 1, pad: int = 0, dilation: int = 1):
        h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
        w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
        return h, w

class Experience(NamedTuple):
    obs: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    next_obs: np.ndarray

Trajectory = List[Experience]

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float):
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha

    def push(self, experience: Experience):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

class Trainer:
    @staticmethod
    def generate_trajectories(env: BaseEnv, q_net: VisualQNetwork, buffer_size: int, epsilon: float):
        buffer: Buffer = []
        env.reset()
        behavior_name = list(env.behavior_specs)[0]
        spec = env.behavior_specs[behavior_name]
        dict_trajectories_from_agent: Dict[int, Trajectory] = {}
        dict_last_obs_from_agent: Dict[int, np.ndarray] = {}
        dict_last_action_from_agent: Dict[int, np.ndarray] = {}
        dict_cumulative_reward_from_agent: Dict[int, float] = {}
        cumulative_rewards: List[float] = []

        while len(buffer) < buffer_size:
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            for agent_id_terminated in terminal_steps:
                last_experience = Experience(
                    obs=dict_last_obs_from_agent[agent_id_terminated].copy(),
                    reward=terminal_steps[agent_id_terminated].reward,
                    done=not terminal_steps[agent_id_terminated].interrupted,
                    action=dict_last_action_from_agent[agent_id_terminated].copy(),
                    next_obs=terminal_steps[agent_id_terminated].obs[0],
                )
                dict_last_obs_from_agent.pop(agent_id_terminated)
                dict_last_action_from_agent.pop(agent_id_terminated)
                cumulative_reward = (
                    dict_cumulative_reward_from_agent.pop(agent_id_terminated)
                    + terminal_steps[agent_id_terminated].reward
                )
                cumulative_rewards.append(cumulative_reward)
                buffer.extend(dict_trajectories_from_agent.pop(agent_id_terminated))
                buffer.append(last_experience)

            for agent_id_decisions in decision_steps:
                if agent_id_decisions not in dict_trajectories_from_agent:
                    dict_trajectories_from_agent[agent_id_decisions] = []
                    dict_cumulative_reward_from_agent[agent_id_decisions] = 0

                if agent_id_decisions in dict_last_obs_from_agent:
                    exp = Experience(
                        obs=dict_last_obs_from_agent[agent_id_decisions].copy(),
                        reward=decision_steps[agent_id_decisions].reward,
                        done=False,
                        action=dict_last_action_from_agent[agent_id_decisions].copy(),
                        next_obs=decision_steps[agent_id_decisions].obs[0],
                    )
                    dict_trajectories_from_agent[agent_id_decisions].append(exp)
                    dict_cumulative_reward_from_agent[agent_id_decisions] += (
                        decision_steps[agent_id_decisions].reward
                    )
                dict_last_obs_from_agent[agent_id_decisions] = (
                    decision_steps[agent_id_decisions].obs[0]
                )

            actions_values = (
                q_net(torch.from_numpy(decision_steps.obs[0])).detach().numpy()
            )
            actions_values += epsilon * (
                np.random.randn(actions_values.shape[0], actions_values.shape[1])
            ).astype(np.float32)
            actions = np.argmax(actions_values, axis=1)
            actions.resize((len(decision_steps), 1))

            for agent_index, agent_id in enumerate(decision_steps.agent_id):
                dict_last_action_from_agent[agent_id] = actions[agent_index]

            action_tuple = ActionTuple()
            action_tuple.add_discrete(actions)
            env.set_actions(behavior_name, action_tuple)
            env.step()
        return buffer, np.mean(cumulative_rewards)

    @staticmethod
    def update_q_net(q_net: VisualQNetwork, target_q_net: VisualQNetwork, optimizer: torch.optim, buffer: PrioritizedReplayBuffer, batch_size: int, action_size: int, beta: float):
        GAMMA = 0.99
        criterion = torch.nn.MSELoss()
        
        batch, indices, weights = buffer.sample(batch_size, beta)
        weights = torch.from_numpy(weights).float()
        
        obs = torch.from_numpy(np.stack([ex.obs for ex in batch])).float()
        reward = torch.from_numpy(np.array([ex.reward for ex in batch], dtype=np.float32).reshape(-1, 1))
        done = torch.from_numpy(np.array([ex.done for ex in batch], dtype=np.float32).reshape(-1, 1))
        action = torch.from_numpy(np.stack([ex.action for ex in batch]))
        next_obs = torch.from_numpy(np.stack([ex.next_obs for ex in batch])).float()

        target = reward + (1.0 - done) * GAMMA * torch.max(target_q_net(next_obs).detach(), dim=1, keepdim=True).values
        mask = torch.ones((len(batch), action_size))
        #mask.scatter_(1, action, 1)
        #prediction = q_net(obs)
        prediction = torch.sum(q_net(obs) * mask, dim=1, keepdim=True)
        #loss = (criterion(prediction, target)).mean()
        #loss = torch.nn.functional.mse_loss(prediction, target)
        loss = (criterion(prediction, target) * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update priorities
        td_errors = (target - prediction).detach().numpy()
        new_priorities = np.abs(td_errors) + 1e-6
        buffer.update_priorities(indices, new_priorities)

        return loss.item()

    @staticmethod
    def soft_update(target_net: torch.nn.Module, source_net: torch.nn.Module, tau: float):
      for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def main():
    env = default_registry["GridWorld"].make()
    num_actions = 5
    q_net = VisualQNetwork((3, 64, 84), 126, num_actions)
    target_q_net = VisualQNetwork((3, 64, 84), 126, num_actions)
    target_q_net.load_state_dict(q_net.state_dict())
    replay_buffer = PrioritizedReplayBuffer(10000, alpha=0.6)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=0.001)
    cumulative_rewards: List[float] = []

    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    beta_start = 0.4
    beta_frames = 1000
    beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
    
    batch_size = 64
    tau = 0.005  # Soft update coefficient
    num_training_steps = 2000
    num_new_exp = 1000

    # Setup TensorBoard
    writer = SummaryWriter(log_dir="C:/Users/minna/Documents/Repository/tensorboard/")

    for step in range(num_training_steps):
        new_experiences, _ = Trainer.generate_trajectories(env, q_net, num_new_exp, epsilon)
        for exp in new_experiences:
            replay_buffer.push(exp)
        
        if len(replay_buffer) > batch_size:
            beta = beta_by_frame(step)
            loss = Trainer.update_q_net(q_net, target_q_net, optimizer, replay_buffer, batch_size, num_actions, beta)
            writer.add_scalar("Loss", loss, step)
        
        # Perform soft update
        Trainer.soft_update(target_q_net, q_net, tau)
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        _, rewards = Trainer.generate_trajectories(env, q_net, 100, epsilon)
        cumulative_rewards.append(rewards)
        writer.add_scalar("Reward", rewards, step)
        writer.add_scalar("Epsilon", epsilon, step)
        print("Training step ", step+1, "\treward ", rewards)

        if(step % 50 == 0):
            target_q_net.load_state_dict(q_net.state_dict())

    env.close()
    writer.close()

if __name__ == '__main__':
    main()
