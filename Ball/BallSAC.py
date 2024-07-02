from stable_baselines3 import SAC
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

def main():
  unity_env = UnityEnvironment("Ball")
  env = UnityToGymWrapper(unity_env, uint8_visual=True,flatten_branched = True)
  tmp_path = "C:/Users/minna/Documents/Repository/tensorboard/Ball/"

  model = SAC("MlpPolicy",
    env,
    verbose=1,
    tensorboard_log = tmp_path)
  
  model.learn(total_timesteps=50000)
  
  env.close()
  print("Saving model to unity_model.zip")
  model.save("unity_model.zip")


if __name__ == '__main__':
  main()