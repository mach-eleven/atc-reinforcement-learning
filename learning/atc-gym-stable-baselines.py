import datetime
import os
import sys
import uuid
from typing import Callable, Optional
from multiprocessing import freeze_support

import gymnasium as gym
import numpy as np
from gymnasium import logger
from gymnasium.wrappers import TransformReward
from gymnasium.wrappers.monitoring import video_recorder
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder
from torch.optim.lr_scheduler import LinearLR

sys.path.append('/home/dev/Projects/lab/atc-reinforcement-learning/atc')
import envs.atc.atc_gym


def capped_cubic_video_schedule(episode_id: int) -> bool:
    """The default episode trigger.

    This function will trigger recordings at the episode indices 0, 1, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000, ...

    Args:
        episode_id: The episode number

    Returns:
        If to apply a video schedule number
    """
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0



class RecordVideo(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper records videos of rollouts.

    Usually, you only want to record episodes intermittently, say every hundredth episode.
    To do this, you can specify **either** ``episode_trigger`` **or** ``step_trigger`` (not both).
    They should be functions returning a boolean that indicates whether a recording should be started at the
    current episode or step, respectively.
    If neither :attr:`episode_trigger` nor ``step_trigger`` is passed, a default ``episode_trigger`` will be employed.
    By default, the recording will be stopped once a `terminated` or `truncated` signal has been emitted by the environment. However, you can
    also create recordings of fixed length (possibly spanning several episodes) by passing a strictly positive value for
    ``video_length``.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        disable_logger: bool = False,
    ):
        """Wrapper records videos of rollouts.

        Args:
            env: The environment that will be wrapped
            video_folder (str): The folder where the recordings will be stored
            episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
            step_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this step
            video_length (int): The length of recorded episodes. If 0, entire episodes are recorded.
                Otherwise, snippets of the specified length are captured
            name_prefix (str): Will be prepended to the filename of the recordings
            disable_logger (bool): Whether to disable moviepy logger or not.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
            disable_logger=disable_logger,
        )
        gym.Wrapper.__init__(self, env)

        if env.render_mode in {None, "human", "ansi", "ansi_list"}:
            raise ValueError(
                f"Render mode is {env.render_mode}, which is incompatible with"
                f" RecordVideo. Initialize your environment with a render_mode"
                f" that returns an image, such as rgb_array."
            )

        if episode_trigger is None and step_trigger is None:
            episode_trigger = capped_cubic_video_schedule

        trigger_count = sum(x is not None for x in [episode_trigger, step_trigger])
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.video_recorder: Optional[video_recorder.VideoRecorder] = None
        self.disable_logger = disable_logger

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder ",
                f"(try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.terminated = False
        self.truncated = False
        self.recorded_frames = 0
        self.episode_id = 0

        try:
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.is_vector_env = False

    def reset(self, **kwargs):
        """Reset the environment using kwargs and then starts recording if video enabled."""
        observations = super().reset(**kwargs)
        self.terminated = False
        self.truncated = False
        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.recorded_frames = []
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames > self.video_length:
                    self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder()
        return observations

    def start_video_recorder(self):
        """Starts video recorder using :class:`video_recorder.VideoRecorder`."""
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=base_path,
            metadata={"step_id": self.step_id, "episode_id": self.episode_id},
            disable_logger=self.disable_logger,
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        else:
            return self.episode_trigger(self.episode_id)

    def step(self, action):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)

        if not (self.terminated or self.truncated):
            # increment steps and episodes
            self.step_id += 1
            if not self.is_vector_env:
                if terminateds or truncateds:
                    self.episode_id += 1
                    self.terminated = terminateds
                    self.truncated = truncateds
            elif terminateds[0] or truncateds[0]:
                self.episode_id += 1
                self.terminated = terminateds[0]
                self.truncated = truncateds[0]

            if self.recording:
                assert self.video_recorder is not None
                self.video_recorder.capture_frame()
                self.recorded_frames += 1
                if self.video_length > 0:
                    if self.recorded_frames > self.video_length:
                        self.close_video_recorder()
                else:
                    if not self.is_vector_env:
                        if terminateds or truncateds:
                            self.close_video_recorder()
                    elif terminateds[0] or truncateds[0]:
                        self.close_video_recorder()

            elif self._video_enabled():
                self.start_video_recorder()

        return observations, rewards, terminateds, truncateds, infos

    def close_video_recorder(self):
        """Closes the video recorder if currently recording."""
        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 1

    def render(self, *args, **kwargs):
        """Compute the render frames as specified by render_mode attribute during initialization of the environment or as specified in kwargs."""
        if self.video_recorder is None or not self.video_recorder.enabled:
            return super().render(*args, **kwargs)

        if len(self.video_recorder.render_history) > 0:
            recorded_frames = [
                self.video_recorder.render_history.pop()
                for _ in range(len(self.video_recorder.render_history))
            ]
            if self.recording:
                return recorded_frames
            else:
                return recorded_frames + super().render(*args, **kwargs)
        else:
            return super().render(*args, **kwargs)

    def close(self):
        """Closes the wrapper then the video recorder."""
        super().close()


class Schedule(object):
    def value(self, step):
        """
        Value of the schedule for a given timestep

        :param step: (int) the timestep
        :return: (float) the output value for the given timestep
        """
        raise NotImplementedError


class LinearSchedule(Schedule):
    """
    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.

    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, step):
        fraction = min(float(step) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


    def constfn(self, val):
        """
        Create a function that returns a constant
        It is useful for learning rate schedule (to avoid code duplication)

        :param val: (float)
        :return: (function)
        """

        def func(_):
            return val

        return func
    

    def get_schedule_fn(self, value_schedule):
        """
        Transform (if needed) learning rate and clip range
        to callable.

        :param value_schedule: (callable or float)
        :return: (function)
        """
        # If the passed schedule is a float
        # create a constant function
        if isinstance(value_schedule, (float, int)):
            # Cast to float to avoid errors
            value_schedule = self.constfn(float(value_schedule))
        else:
            assert callable(value_schedule)
        return value_schedule


class ModelFactory:
    hyperparams: dict

    def build(self, env, log_dir):
        pass


class PPOModelFactory(ModelFactory):
    def __init__(self):
        self.hyperparams = {"n_steps": 1024,
                            "batch_size": 32,
                            "clip_range": 0.4,
                            "gamma": 0.996,
                            "gae_lambda": 0.95,
                            # "learning_rate": LinearLR(),
                            "learning_rate": LinearSchedule(1.0, initial_p=0.0002, final_p=0.001).value,
                            "n_epochs": 4,
                            "ent_coef": 0.002}

    def build(self, env):
        return PPO("MlpPolicy", env, verbose=1, **self.hyperparams)


def learn(model_factory: ModelFactory, multiprocess: bool = True, time_steps: int = int(1e6),
          record_video: bool = True):
    def callback(locals_, globals_):
        self_ = locals_["self"]

        # mean_actions = np.mean(self_.env.get_attr("actions_per_timestep"))
        # print("mean_actions: ", mean_actions)
        # mean_actions_tf = tf.Summary(value=[tf.Summary.Value(tag='simulation/mean_actions', simple_value=mean_actions)])
        # winning_ratio = np.mean(self_.env.get_attr("winning_ratio"))
        # winning_ratio_tf = tf.Summary(
        #     value=[tf.Summary.Value(tag='simulation/winning_ratio', simple_value=winning_ratio)])
        # locals_['writer'].add_summary(mean_actions_tf, self_.num_timesteps)
        # locals_['writer'].add_summary(winning_ratio_tf, self_.num_timesteps)

        # if isinstance(model_factory, PPOModelFactory):
        #     fps = tf.Summary(value=[tf.Summary.Value(tag='simulation/fps', simple_value=locals_['fps'])])
        #     mean_length = np.mean([info["l"] for info in locals_["ep_infos"]])
        #     mean_length_tf = tf.Summary(
        #         value=[tf.Summary.Value(tag='simulation/mean_episode_length', simple_value=mean_length)])
        #     locals_['writer'].add_summary(fps, self_.num_timesteps)
        #     locals_['writer'].add_summary(mean_length_tf, self_.num_timesteps)

        return True

    def video_trigger(step):
        if not record_video or step < time_steps / 3:
            return False

        return step % (int(time_steps / 8)) == 0

    print(datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))

    

    def make_env():
        log_dir_single = f"{base_dir}/{uuid.uuid4()}/"
        env = gym.make('AtcEnv-v0')

        os.makedirs(log_dir_single, exist_ok=True)
        env = Monitor(env, log_dir_single, allow_early_resets=True)

        return env

    n_envs = 8
    if multiprocess:
        env = SubprocVecEnv([lambda: make_env() for i in range(n_envs)])
    else:
        env = DummyVecEnv([lambda: make_env()])

    if record_video:
        env = VecVideoRecorder(env, video_dir, video_trigger, video_length=2000)

    model = model_factory.build(env)

    # Saving model hyperparams 
    # yaml.dump(model_factory.hyperparams, open(os.path.join(model_dir, "hyperparams.yml"), "w+"))

    model_number = 0
    while True:
        model = model_factory.build(env)
        model.learn(total_timesteps=820000, callback=callback)
        model_number += 1

        is_won, new_total_reward = play_to_train(model)

        print(f"model number #{model_number} with {new_total_reward} points!")

        if is_won:
            model.save(f"{model_dir}/PPO_atc_gym_won8")
            break

# def play_to_train(model):
#     total_reward = 0
    
#     new_env = gym.make('AtcEnv-v0')
#     obs = new_env.reset()[0]
#     while True:
#         action, _states = model.predict(obs)
#         obs, rewards, done, truncated, info = new_env.step(action)

#         total_reward += rewards

#         if rewards > 1:
#             print(rewards)
#             print('Next to corridor!')

#         new_env.render()
#         if done:
#             print(total_reward)
#             new_env.close()
#             if rewards > 10000:
#                 return (True, total_reward)
#             else:
#                 return (False, total_reward)


def play_to_train(model):
    total_reward = 0
    
    new_env = gym.make('AtcEnv-v0')
    obs = new_env.reset()[0]
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, truncated, info = new_env.step(action)

        total_reward += rewards

        new_env.render()
        if done:
            print(total_reward)
            new_env.close()
            print(rewards)
            if rewards > 10000:
                return (True, total_reward)
            else:
                return (False, total_reward)
            

def play_to_win():
    total_reward = 0

    model = PPO.load(f"{model_dir}/PPO_atc_gym_won8.zip", device="cpu")
    new_env = gym.make('AtcEnv-v0')
    # new_env = TransformReward(new_env, lambda r: np.random.uniform(low=-3, high=3))
    obs = new_env.reset()[0]

    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, truncated, info = new_env.step(action)

        total_reward += rewards

        new_env.render()
        if done:
            obs = new_env.reset()[0]

            # # new_env.close()
            # if rewards > 10000:
            #     # print('Won!')
            #     return total_reward, 1
            # else:
            #     # print('Lost!')
            #     return total_reward, 0
            # break


def play_to_attack():
    total_reward = 0

    model = PPO.load(f"{model_dir}/PPO_atc_gym_won8.zip", device="cpu")
    new_env = gym.make('AtcEnv-v0')
    # new_env = TransformReward(new_env, lambda r: np.random.uniform(low=-3, high=3))
    obs = new_env.reset()[0]

    while True:
        action, _states = model.predict(obs)

        if np.random.choice([0, 1]):
            action = [np.random.uniform(low=-1, high=1), np.random.uniform(low=-1, high=1), np.random.uniform(low=-1, high=1)]

        obs, rewards, done, truncated, info = new_env.step(action)

        total_reward += rewards
        # print(f"reward: {rewards}")

        new_env.render()
        if done:
            # print(f"Last rewards: {rewards}")
            new_env.close()
            if rewards > 10000:
                # print('Won!')
                return total_reward, 1
            else:
                # print('Lost!')
                return total_reward, 0
            break


if __name__ == '__main__':
    freeze_support()
    base_dir = ".."

    model_dir = os.path.join("../", "model")
    os.makedirs(model_dir, exist_ok=True)

    video_dir = os.path.join("../", "videos")
    os.makedirs(video_dir, exist_ok=True)

    # learn(PPOModelFactory(), time_steps=820000, multiprocess=True, record_video=False)

    win_count = {'attack': 0, 'original': 0}
    total_score = {'attack': 0, 'original': 0}

    # for episode in range(10):
    #     score, is_win = play_to_attack()
    #     total_score["attack"] += score 
    #     win_count["attack"] += is_win 
        

    #     score, is_win = play_to_win()
    #     total_score["original"] += score 
    #     win_count["original"] += is_win

    #     print(win_count, total_score)

    # print(win_count, total_score)

    play_to_win()


    # model = PPO.load(f"{model_dir}/PPO_atc_gym_won8.zip", device="cpu")
    # new_env = gym.make('AtcEnv-v0')
    # # new_env = TransformReward(new_env, lambda r: np.random.uniform(low=-3, high=3))
    
    # new_env = RecordVideo(new_env, '..')

    # new_env.reset()
    # new_env.render()

    # new_env.start_video_recorder()

    # total_reward = 0

    # model = PPO.load(f"{model_dir}/PPO_atc_gym_won8.zip", device="cpu")

    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, done, truncated, info = new_env.step(action)

    #     total_reward += rewards

    #     new_env.render()
    #     if done:
    #         new_env.close()
    #         if rewards > 10000:
    #             break
    #         else:
    #             break
    #         break

    # new_env.close_video_recorder()

    # # Close the environment
    # new_env.close()