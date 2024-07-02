import gym

from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper


def main():
  unity_env = UnityEnvironment("GridWorld")
  env = UnityToGymWrapper(unity_env, uint8_visual=True)
  tmp_path = "C:/Users/minna/Documents/Repository/tensorboard/"
  #new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
 
 
  model = A2C("MlpPolicy", env, verbose=1,tensorboard_log = tmp_path)
  model.learn(total_timesteps=30000)
  
  env.close()
  model.save("unity_model_ppo.zip")


if __name__ == '__main__':
  main()