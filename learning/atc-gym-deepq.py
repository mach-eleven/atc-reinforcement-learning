import sys
import time

import gymnasium as gym
from stable_baselines3 import DQN

# from baselines import deepq

sys.path.append('..')

# noinspection PyUnresolvedReferences (is used for registering the atc gym in the OpenAI gym framework)
import envs.atc.atc_gym


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) >= 20
    return is_solved


def main():
    env = gym.make('AtcEnv-v0')
    env.reset()
    
    model = DQN(policy="MlpPolicy", env=env, learning_rate=1e-3, buffer_size=50000, 
                exploration_fraction=0.1, exploration_final_eps=0.02, train_freq=10)

    act = model.learn(
        total_timesteps=100000,
        callback=callback
    )

    print("Saving model")
    act.save("atc-gym-deepq.pkl")


if __name__ == '__main__':
    main()