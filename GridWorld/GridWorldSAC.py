import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, NamedTuple, Dict
from math import floor
from torch.nn import Parameter
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.registry import default_registry
from mlagents_envs.environment import ActionTuple, BaseEnv

# Experience tuple
class Experience(NamedTuple):
    obs: np.ndarray
    additional_obs: np.ndarray
    action: int  # Ensure action is an integer for discrete actions
    reward: float
    done: bool
    next_obs: np.ndarray

Trajectory = List[Experience]

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.position = 0

    def push(self, experience: Experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        return samples

    def __len__(self):
        return len(self.buffer)

# Neural Networks
class VisualQNetwork(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], encoding_size: int, output_size: int):
        super(VisualQNetwork, self).__init__()
        height = input_shape[1]
        width = input_shape[2]
        initial_channels = input_shape[0]
        conv_1_hw = self.conv_output_shape((height, width), 8, 4)
        conv_2_hw = self.conv_output_shape(conv_1_hw, 4, 2)
        self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32
        self.conv1 = nn.Conv2d(initial_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        
        self.fc1 = nn.Linear(self.final_flat, encoding_size)
        self.fc2 = nn.Linear(encoding_size + 2, encoding_size)  # Adjust based on additional obs

        self.output = nn.Linear(encoding_size, output_size)

    def forward(self, visual_obs: torch.tensor, additional_obs: torch.tensor):
        conv_1 = torch.nn.functional.leaky_relu(self.conv1(visual_obs), 0.01)
        conv_2 = torch.nn.functional.leaky_relu(self.conv2(conv_1), 0.01)
        flattened = conv_2.reshape([-1, self.final_flat])
        fc1_out = torch.nn.functional.leaky_relu(self.fc1(flattened), 0.01)
        combined = torch.cat((fc1_out, additional_obs), dim=1)
        fc2_out = torch.nn.functional.leaky_relu(self.fc2(combined), 0.01)
        return self.output(fc2_out)

    @staticmethod
    def conv_output_shape(h_w: Tuple[int, int], kernel_size: int = 1, stride: int = 1, pad: int = 0, dilation: int = 1):
        h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
        w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
        return h, w

class PolicyNetwork(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], encoding_size: int, action_size: int):
        super(PolicyNetwork, self).__init__()
        height = input_shape[1]
        width = input_shape[2]
        initial_channels = input_shape[0]
        conv_1_hw = self.conv_output_shape((height, width), 8, 4)
        conv_2_hw = self.conv_output_shape(conv_1_hw, 4, 2)
        self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32
        self.conv1 = nn.Conv2d(initial_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        
        self.fc1 = nn.Linear(self.final_flat, encoding_size)
        self.fc2 = nn.Linear(encoding_size + 2, encoding_size)  # Adjust based on additional obs

        self.logits = nn.Linear(encoding_size, action_size)

    def forward(self, visual_obs: torch.tensor, additional_obs: torch.tensor):
        conv_1 = torch.nn.functional.leaky_relu(self.conv1(visual_obs), 0.01)
        conv_2 = torch.nn.functional.leaky_relu(self.conv2(conv_1), 0.01)
        flattened = conv_2.reshape([-1, self.final_flat])
        fc1_out = torch.nn.functional.leaky_relu(self.fc1(flattened), 0.01)
        combined = torch.cat((fc1_out, additional_obs), dim=1)
        fc2_out = torch.nn.functional.leaky_relu(self.fc2(combined), 0.01)
        logits = self.logits(fc2_out)
        return logits

    @staticmethod
    def conv_output_shape(h_w: Tuple[int, int], kernel_size: int = 1, stride: int = 1, pad: int = 0, dilation: int = 1):
        h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
        w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
        return h, w

class Trainer:
    def __init__(self, env, buffer_size, batch_size, gamma, tau, alpha, lr):
        self.env = env
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        num_actions = 5
        input_shape = (3, 64, 84)
        encoding_size = 126

        self.q_net1 = VisualQNetwork(input_shape, encoding_size, num_actions)
        self.q_net2 = VisualQNetwork(input_shape, encoding_size, num_actions)
        self.target_q_net1 = VisualQNetwork(input_shape, encoding_size, num_actions)
        self.target_q_net2 = VisualQNetwork(input_shape, encoding_size, num_actions)
        self.policy_net = PolicyNetwork(input_shape, encoding_size, num_actions)

        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.writer = SummaryWriter(log_dir="C:/Users/minna/Documents/Repository/tensorboard/SAC")

    def update(self, step):
        experiences = self.buffer.sample(self.batch_size)
        obs = torch.FloatTensor([exp.obs for exp in experiences])
        additional_obs = torch.FloatTensor([exp.additional_obs for exp in experiences])
        action = torch.LongTensor([[exp.action] for exp in experiences])
        reward = torch.FloatTensor([exp.reward for exp in experiences])
        done = torch.FloatTensor([exp.done for exp in experiences])
        next_obs = torch.FloatTensor([exp.next_obs for exp in experiences])

        with torch.no_grad():
            next_logits = self.policy_net(next_obs, additional_obs)
            next_action_probs = torch.nn.functional.softmax(next_logits, dim=-1)
            next_action_dist = torch.distributions.Categorical(next_action_probs)
            next_action = next_action_dist.sample()
            next_log_prob = next_action_dist.log_prob(next_action).unsqueeze(-1)
            next_q_value = torch.min(
                self.target_q_net1(next_obs, additional_obs),
                self.target_q_net2(next_obs, additional_obs)
            ).gather(1, next_action.unsqueeze(-1))
            next_q_value = next_q_value - self.alpha * next_log_prob
            target_q_value = reward + (1 - done) * self.gamma * next_q_value.squeeze(-1)

        q_value1 = self.q_net1(obs, additional_obs).gather(1, action).squeeze(-1)
        q_value2 = self.q_net2(obs, additional_obs).gather(1, action).squeeze(-1)
        q_loss1 = nn.MSELoss()(q_value1, target_q_value)
        q_loss2 = nn.MSELoss()(q_value2, target_q_value)

        self.q_optimizer1.zero_grad()
        q_loss1.backward()
        self.q_optimizer1.step()

        self.q_optimizer2.zero_grad()
        q_loss2.backward()
        self.q_optimizer2.step()

        logits = self.policy_net(obs, additional_obs)
        action_probs = torch.nn.functional.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        new_action = action_dist.sample()
        log_prob = action_dist.log_prob(new_action).unsqueeze(-1)
        q_value = torch.min(
            self.q_net1(obs, additional_obs),
            self.q_net2(obs, additional_obs)
        ).gather(1, new_action.unsqueeze(-1)).squeeze(-1)
        policy_loss = (self.alpha * log_prob.squeeze(-1) - q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.writer.add_scalar("Loss/Q1", q_loss1.item(), step)
        self.writer.add_scalar("Loss/Q2", q_loss2.item(), step)
        self.writer.add_scalar("Loss/Policy", policy_loss.item(), step)

    def soft_update(self, target_net, source_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def generate_trajectories(self, buffer_size: int, epsilon: float):
        buffer: List[Experience] = []
        self.env.reset()
        behavior_name = list(self.env.behavior_specs)[0]
        dict_trajectories_from_agent: Dict[int, Trajectory] = {}
        dict_last_obs_from_agent: Dict[int, np.ndarray] = {}
        dict_last_action_from_agent: Dict[int, int] = {}  # Change to int
        dict_cumulative_reward_from_agent: Dict[int, float] = {}
        cumulative_rewards: List[float] = []

        while len(buffer) < buffer_size:
            decision_steps, terminal_steps = self.env.get_steps(behavior_name)

            for agent_id_terminated in terminal_steps:
                last_experience = Experience(
                    obs=dict_last_obs_from_agent[agent_id_terminated].copy(),
                    additional_obs=terminal_steps[agent_id_terminated].obs[1],
                    reward=terminal_steps[agent_id_terminated].reward,
                    done=not terminal_steps[agent_id_terminated].interrupted,
                    action=dict_last_action_from_agent[agent_id_terminated],  # Remove .item()
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
                        additional_obs=decision_steps[agent_id_decisions].obs[1],
                        reward=decision_steps[agent_id_decisions].reward,
                        done=False,
                        action=dict_last_action_from_agent[agent_id_decisions],  # Remove .item()
                        next_obs=decision_steps[agent_id_decisions].obs[0],
                    )
                    dict_trajectories_from_agent[agent_id_decisions].append(exp)
                    dict_cumulative_reward_from_agent[agent_id_decisions] += (
                        decision_steps[agent_id_decisions].reward
                    )
                dict_last_obs_from_agent[agent_id_decisions] = (
                    decision_steps[agent_id_decisions].obs[0]
                )

            visual_obs = torch.from_numpy(decision_steps.obs[0])
            additional_obs = torch.from_numpy(decision_steps.obs[1])
            logits = self.policy_net(visual_obs, additional_obs)
            action_probs = torch.nn.functional.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            actions = action_dist.sample()

            for agent_index, agent_id in enumerate(decision_steps.agent_id):
                dict_last_action_from_agent[agent_id] = actions[agent_index].item()

            action_tuple = ActionTuple()
            action_tuple.add_discrete(actions.unsqueeze(-1).numpy())
            self.env.set_actions(behavior_name, action_tuple)
            self.env.step()
        return buffer, np.mean(cumulative_rewards)

def main():
    env = default_registry["GridWorld"].make()
    buffer_size = 10000
    batch_size = 64
    gamma = 0.99
    tau = 0.005
    alpha = 0.2
    lr = 0.001
    num_training_steps = 10000
    num_new_exp = 1000
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995

    trainer = Trainer(env, buffer_size, batch_size, gamma, tau, alpha, lr)

    for step in range(num_training_steps):
        new_experiences, rewards = trainer.generate_trajectories(num_new_exp, epsilon)
        for exp in new_experiences:
            trainer.buffer.push(exp)

        if len(trainer.buffer) > batch_size:
            trainer.update(step)
            trainer.soft_update(trainer.target_q_net1, trainer.q_net1)
            trainer.soft_update(trainer.target_q_net2, trainer.q_net2)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Log and print rewards
        trainer.writer.add_scalar("Reward", rewards, step)
        print(f"Training step {step + 1} \t Reward: {rewards}")

    env.close()
    trainer.writer.close()

if __name__ == '__main__':
    main()
