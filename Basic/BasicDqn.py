import gym

from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
import torch.nn as nn
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



def main():
  unity_env = UnityEnvironment("Basic")
  env = UnityToGymWrapper(unity_env, uint8_visual=True,flatten_branched = True)
  tmp_path = "C:/Users/minna/Documents/Repository/tensorboard/Basic/"
  #new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
  #env = VecNormalize(env, norm_reward=True)

  
  model = DQN("MlpPolicy",
    env,
    learning_rate= 0.0005,
    buffer_size=20000,
    batch_size=256,
    gamma=0.99,
    exploration_fraction=1,
    exploration_final_eps=0.01,
    target_update_interval=500,
    train_freq=4,
    gradient_steps=8,
    verbose=1,
    tensorboard_log = tmp_path)
  
  model.learn(total_timesteps=50000)
  
  env.close()
  print("Saving model to unity_model.zip")
  model.save("unity_model.zip")


if __name__ == '__main__':
  main()