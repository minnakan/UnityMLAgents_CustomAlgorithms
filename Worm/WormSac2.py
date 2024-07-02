import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from collections import namedtuple, deque
import random
from math import pi as pi_constant
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


# SAC class
Tensor = torch.DoubleTensor
torch.set_default_tensor_type(Tensor)

class SAC(nn.Module):
    def __init__(self, config):
        super(SAC, self).__init__()
        torch.manual_seed(config['seed'])

        self.lr = config['lr']  # learning rate
        self.smooth = config['smooth']  # smoothing coefficient for target net
        self.discount = config['discount']  # discount factor
        self.alpha = config['alpha']  # temperature parameter in SAC
        self.batch_size = config['batch_size']  # mini batch size

        self.dims_hidden_neurons = config['dims_hidden_neurons']
        self.dim_obs = config['dim_obs']
        self.dim_action = config['dim_action']

        self.device = config['device']

        self.actor = ActorNet(device=config['device'],
                              dim_obs=self.dim_obs,
                              dim_action=self.dim_action,
                              dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)
        self.Q1 = QCriticNet(dim_obs=self.dim_obs,
                             dim_action=self.dim_action,
                             dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)
        self.Q2 = QCriticNet(dim_obs=self.dim_obs,
                             dim_action=self.dim_action,
                             dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)
        self.Q1_tar = QCriticNet(dim_obs=self.dim_obs,
                                 dim_action=self.dim_action,
                                 dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)
        self.Q2_tar = QCriticNet(dim_obs=self.dim_obs,
                                 dim_action=self.dim_action,
                                 dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_Q1 = optim.Adam(self.Q1.parameters(), lr=self.lr)
        self.optimizer_Q2 = optim.Adam(self.Q2.parameters(), lr=self.lr)

    def update(self, buffer, writer, global_step):
        # sample from replay memory
        t = buffer.sample(self.batch_size)
        states = torch.tensor(t.obs, dtype=torch.double).to(self.device)
        actions = torch.tensor(t.action, dtype=torch.double).to(self.device)
        rewards = torch.tensor(t.reward, dtype=torch.double).to(self.device)
        next_states = torch.tensor(t.next_obs, dtype=torch.double).to(self.device)
        dones = torch.tensor(t.done, dtype=torch.float).to(self.device)

        # Perform the updates for the actor and critic networks
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor(next_states)
            q1_next_target = self.Q1_tar(next_states, next_state_action)
            q2_next_target = self.Q2_tar(next_states, next_state_action)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + (1.0 - dones) * self.discount * min_q_next_target

        q1_value = self.Q1(states, actions)
        q2_value = self.Q2(states, actions)
        q1_loss = nn.MSELoss()(q1_value, next_q_value)
        q2_loss = nn.MSELoss()(q2_value, next_q_value)

        self.optimizer_Q1.zero_grad()
        q1_loss.backward()
        self.optimizer_Q1.step()

        self.optimizer_Q2.zero_grad()
        q2_loss.backward()
        self.optimizer_Q2.step()

        # Log critic losses
        writer.add_scalar('Loss/Q1', q1_loss.item(), global_step)
        writer.add_scalar('Loss/Q2', q2_loss.item(), global_step)

        # Update Actor network
        pi, log_pi, _ = self.actor(states)
        q1_pi = self.Q1(states, pi)
        q2_pi = self.Q2(states, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Log actor loss
        writer.add_scalar('Loss/Actor', actor_loss.item(), global_step)

        # Update target networks
        for target_param, param in zip(self.Q1_tar.parameters(), self.Q1.parameters()):
            target_param.data.copy_(
                self.smooth * param.data + (1.0 - self.smooth) * target_param.data
            )
        for target_param, param in zip(self.Q2_tar.parameters(), self.Q2.parameters()):
            target_param.data.copy_(
                self.smooth * param.data + (1.0 - self.smooth) * target_param.data
            )

    def act_probabilistic(self, obs: torch.Tensor):
        self.actor.eval()
        a, logProb, mu = self.actor(obs)
        self.actor.train()
        return a

    def act_deterministic(self, obs: torch.Tensor):
        self.actor.eval()
        a, logProb, mu = self.actor(obs)
        self.actor.train()
        return mu


class ActorNet(nn.Module):
    def __init__(self,
                 device,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int] = (64, 64)
                 ):
        super(ActorNet, self).__init__()
        self.n_layers = len(dims_hidden_neurons)
        self.dim_action = dim_action
        self.device = device

        self.ln2pi = torch.log(Tensor([2*pi_constant]))

        n_neurons = (dim_obs,) + dims_hidden_neurons + (dim_action,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))

        self.output_mu = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output_mu.weight)
        torch.nn.init.zeros_(self.output_mu.bias)

        self.output_logsig = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output_logsig.weight)
        torch.nn.init.zeros_(self.output_logsig.bias)

    def forward(self, obs: torch.Tensor):
        x = obs
        for i in range(self.n_layers):
            x = eval('torch.relu(self.layer{}(x))'.format(i + 1))
        mu = self.output_mu(x)
        sig = torch.exp(self.output_logsig(x))

        u = mu + sig * torch.normal(torch.zeros(size=mu.shape), 1).to(self.device)
        a = torch.tanh(u)
        logProbu = -1/2 * (torch.sum(torch.log(sig**2), dim=1, keepdims=True).to(self.device) +
                           torch.sum((u-mu)**2/sig**2, dim=1, keepdims=True) +
                           a.shape[1]*self.ln2pi.to(self.device))
        logProba = logProbu - torch.sum(torch.log(1 - a ** 2 + 0.000001), dim=1, keepdims=True)
        return a, logProba, torch.tanh(mu)


class QCriticNet(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int] = (64, 64)):
        super(QCriticNet, self).__init__()
        self.n_layers = len(dims_hidden_neurons)
        self.dim_action = dim_action

        n_neurons = (dim_obs + dim_action,) + dims_hidden_neurons + (1,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat((obs, action), dim=1)
        for i in range(self.n_layers):
            x = eval('torch.relu(self.layer{}(x))'.format(i + 1))
        return self.output(x)


def main():
    unity_env = UnityEnvironment("Worm")
    env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=True)
    
    config = {
        'seed': 0,
        'lr': 3e-4,
        'smooth': 0.005,
        'discount': 0.99,
        'alpha': 0.2,
        'batch_size': 64,
        'dims_hidden_neurons': (256, 256),
        'dim_obs': env.observation_space.shape[0],
        'dim_action': env.action_space.shape[0],
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    model = SAC(config)
    replay_buffer = ReplayBuffer(capacity=1000000)
    writer = SummaryWriter(log_dir="C:/Users/minna/Documents/Repository/tensorboard/Worm/")

    num_episodes = 1000
    max_timesteps = 50000
    total_timesteps = 0
    global_step = 0

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0

        for t in range(max_timesteps):
            action = model.act_probabilistic(torch.tensor(obs, dtype=torch.double).unsqueeze(0).to(config['device'])).detach().cpu().numpy().flatten()
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.push(obs, action, next_obs, reward, done)
            obs = next_obs
            episode_reward += reward
            total_timesteps += 1
            global_step += 1

            if len(replay_buffer) > config['batch_size']:
                model.update(replay_buffer, writer, global_step)

            if done:
                break

        writer.add_scalar('Reward/Episode', episode_reward, episode)
        print(f"Episode {episode + 1}: Total Reward: {episode_reward}")


    env.close()
    writer.close()
    print("Saving model to unity_model.zip")
    torch.save(model.state_dict(), "unity_model.zip")

if __name__ == '__main__':
    main()
