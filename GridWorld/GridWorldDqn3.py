import gym
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
import torch.nn as nn
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomNatureCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super(CustomNatureCNN, self).__init__(observation_space, features_dim=256)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

def main():
    unity_env = UnityEnvironment("GridWorld")
    env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=True)
    tmp_path = "C:/Users/minna/Documents/Repository/tensorboard/"

    observation_space = env.observation_space
    print("Observation space shape:", observation_space.shape)
    
    policy_kwargs = dict(
        features_extractor_class=CustomNatureCNN,
        net_arch=[256, 256]
    )

    model = DQN("CnnPolicy",
                env,
                learning_rate=0.0005,
                buffer_size=20000,
                batch_size=256,
                gamma=0.99,
                exploration_fraction=0.2,
                exploration_final_eps=0.01,
                target_update_interval=500,
                train_freq=4,
                gradient_steps=8,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tmp_path)
    
    model.learn(total_timesteps=50000)
    
    env.close()
    print("Saving model to unity_model.zip")
    model.save("unity_model.zip")

if __name__ == '__main__':
    main()
