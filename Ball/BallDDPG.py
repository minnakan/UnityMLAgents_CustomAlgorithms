import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from collections import namedtuple, deque
import random
from typing import Tuple

# Replay buffer
Transition = namedtuple('Transition', ('obs', 'action', 'next_obs', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.buffer)


# DDPG class
class DDPG(nn.Module):
    def __init__(self, config):
        super(DDPG, self).__init__()
        torch.manual_seed(config['seed'])

        self.lr = config['lr']
        self.smooth = config['smooth']
        self.discount = config['discount']
        self.batch_size = config['batch_size']

        self.dim_obs = config['dim_obs']
        self.dim_action = config['dim_action']
        self.dims_hidden_neurons = config['dims_hidden_neurons']

        self.device = config['device']

        self.actor = ActorNet(device=self.device,
                              dim_obs=self.dim_obs,
                              dim_action=self.dim_action,
                              dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)
        self.actor_target = ActorNet(device=self.device,
                                     dim_obs=self.dim_obs,
                                     dim_action=self.dim_action,
                                     dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)
        self.critic = CriticNet(dim_obs=self.dim_obs,
                                dim_action=self.dim_action,
                                dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)
        self.critic_target = CriticNet(dim_obs=self.dim_obs,
                                       dim_action=self.dim_action,
                                       dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.critic_loss = nn.MSELoss()

    def update(self, buffer, writer, global_step):
        if len(buffer) < self.batch_size:
            return

        # sample from replay memory
        t = buffer.sample(self.batch_size)
        states = torch.tensor(t.obs, dtype=torch.float32).to(self.device)
        actions = torch.tensor(t.action, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(t.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(t.next_obs, dtype=torch.float32).to(self.device)
        dones = torch.tensor(t.done, dtype=torch.float32).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_target = rewards + (1.0 - dones) * self.discount * self.critic_target(next_states, next_actions)
        
        q_value = self.critic(states, actions)
        critic_loss = self.critic_loss(q_value, q_target)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Log critic loss
        writer.add_scalar('Loss/Critic', critic_loss.item(), global_step)

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Log actor loss
        writer.add_scalar('Loss/Actor', actor_loss.item(), global_step)

        # Update target networks
        self.soft_update(self.actor_target, self.actor, self.smooth)
        self.soft_update(self.critic_target, self.critic, self.smooth)

    def soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def act(self, obs, noise=0.0):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = self.actor(obs).cpu().data.numpy().flatten()
        action = action + noise * np.random.randn(self.dim_action)
        return np.clip(action, -1.0, 1.0)


class ActorNet(nn.Module):
    def __init__(self,
                 device,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int] = (400, 300)):
        super(ActorNet, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()

        input_dim = dim_obs
        for hidden_dim in dims_hidden_neurons:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.output_layer = nn.Linear(input_dim, dim_action)
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.output_layer.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs: torch.Tensor):
        x = obs
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = torch.tanh(self.output_layer(x))
        return x


class CriticNet(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int] = (400, 300)):
        super(CriticNet, self).__init__()
        self.layers = nn.ModuleList()

        input_dim = dim_obs + dim_action
        for hidden_dim in dims_hidden_neurons:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.output_layer = nn.Linear(input_dim, 1)
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.output_layer.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=1)
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)


def main():
    unity_env = UnityEnvironment("Ball")
    env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=True)
    
    config = {
        'seed': 0,
        'lr': 1e-4,
        'smooth': 0.005,
        'discount': 0.99,
        'batch_size': 64,
        'dims_hidden_neurons': (400, 300),
        'dim_obs': env.observation_space.shape[0],
        'dim_action': env.action_space.shape[0],
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    model = DDPG(config)
    replay_buffer = ReplayBuffer(capacity=1000000)
    writer = SummaryWriter(log_dir="C:/Users/minna/Documents/Repository/tensorboard/Ball/")

    num_episodes = 1000
    max_timesteps = 50000
    total_timesteps = 0
    global_step = 0

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0

        for t in range(max_timesteps):
            action = model.act(obs, noise=0.1)
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.push(obs, action, next_obs, reward, done)
            obs = next_obs
            episode_reward += reward
            total_timesteps += 1
            global_step += 1

            model.update(replay_buffer, writer, global_step)

            if done:
                break

        writer.add_scalar('Reward/Episode', episode_reward, episode)
        print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

        if total_timesteps >= 5000000:
            break

    env.close()
    writer.close()
    print("Saving model to ddpg_model.zip")
    torch.save(model.state_dict(), "ddpg_model.zip")

if __name__ == '__main__':
    main()
