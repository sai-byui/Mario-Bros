import os
import random

import gym
import gym_super_mario_bros.actions as actions
import numpy as np
import tensorflow as tf
from nes_py.wrappers import JoypadSpace
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.optimizers import Adam
from keras.layers import LeakyReLU
from my_wrappers import MaxAndSkipEnv, FireResetEnv, FrameDownSample, ScaledFloatFrame, LazyFrameStack, CustomReward, \
    ReplyBuffer


# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   tf.config.set_logical_device_configuration(
#     physical_devices[0],
#     tf.config.LogicalDeviceConfiguration(memory_limit=4096))
#
#   logical_devices = tf.config.list_logical_devices('GPU')
#   assert len(logical_devices) == len(physical_devices) + 1
#
#   tf.config.set_logical_device_configuration(
#     physical_devices[0],
#     tf.config.LogicalDeviceConfiguration(memory_limit=4096))
# except:
#   pass


def wrap_nes(env_id, action_space):
    """
    Configure environment for NES.
    :param env_id: (str) the environment ID
    :param action_space: (list) action space
    :return: (Gym Environment) the wrapped environment
    """
    env = gym.make(env_id)
    env = JoypadSpace(env, action_space)
    env = MaxAndSkipEnv(env)

    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = FrameDownSample(env)
    env = ScaledFloatFrame(env)
    env = LazyFrameStack(env, 4)
    env = CustomReward(env)
    return env


class Agent:
    def __init__(self, env, memory_size=20000):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.observation_shape = env.observation_space.shape
        self.memory = ReplyBuffer(memory_size=memory_size)
        self.batch_size = 32
        self.update_frequency = 4
        self.tau = 1000
        self.gamma = 0.99  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9994
        self.learning_rate = 0.005
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        model = Sequential()
        print('observe',self.observation_shape)
        print(('numpy redo',np.reshape(env.reset(), (1,) + env.observation_space.shape)))
        model.add(Conv2D(32, (3, 3), strides=(4, 4), activation=LeakyReLU(alpha=0.05), input_shape=self.observation_shape))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), activation=LeakyReLU(alpha=0.05)))
        model.add(Flatten())
        model.add(Dense(512, activation=LeakyReLU(alpha=0.05), kernel_initializer='random_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_network(self):
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)
        else:
            print('self.model.predict(state)',self.model.predict(state))
            print('self.model.predict(state)[0]',self.model.predict(state)[0])
            print('np.argmax(self.model.predict(state)[0])',np.argmax(self.model.predict(state)[0]))
            return np.argmax(self.model.predict(state)[0])

    def experience_reply(self):
        if self.batch_size > len(self.memory):
            return

        # Get indices of samples for replay buffers
        indices = np.random.choice(range(len(self.memory)), size=self.batch_size)

        # Randomly sample a batch from the memory
        state_sample = np.array([self.memory.state[i][0] for i in indices])
        action_sample = np.array([self.memory.action[i] for i in indices])
        reward_sample = np.array([self.memory.reward[i] for i in indices])
        next_state_sample = np.array([self.memory.next_state[i][0] for i in indices])
        done_sample = np.array([self.memory.done[i] for i in indices])

        # Batch prediction to save speed
        target = self.model.predict(state_sample)
        target_next = self.target_model(next_state_sample)

        for i in range(self.batch_size):
            if done_sample[i]:
                target[i][action_sample[i]] = reward_sample[i]
            else:
                target[i][action_sample[i]] = reward_sample[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(
            np.array(state_sample),
            np.array(target),
            batch_size=self.batch_size,
            verbose=0
        )

    def load_weights(self, weights_file):
        self.epsilon = self.epsilon_min
        self.model.load_weights(weights_file)

    def save_weights(self, weights_file):
        self.model.save_weights(weights_file)


if __name__ == "__main__":
    """
    Main program
    """
    monitors = True

    # Initializes the environment
    env = wrap_nes("SuperMarioBros-1-1-v0", actions.SIMPLE_MOVEMENT)

    # Records the environment
    if monitors:
        env = gym.wrappers.Monitor(env, "recording", video_callable=lambda episode_id: True, force=True)

    # Defines training related constants
    num_episodes = 50000
    num_episode_steps = env.spec.max_episode_steps  # constant value
    frame_count = 0
    max_reward = 0

    # Creates an agent
    agent = Agent(env=env, memory_size=20000)

    # Loads the weights
    if os.path.isfile("super_mario_bros_v0.h5"):
        agent.load_weights("super_mario_bros_v0.h5")

    for episode in range(num_episodes):
        # Defines the total reward per episode
        total_reward = 0

        # Resets the environment
        observation = env.reset()

        # Gets the state
        state = np.reshape(observation, (1,) + env.observation_space.shape)

        for episode_step in range(num_episode_steps):
            # Renders the screen after new environment observation

            # env.render(mode="human")

            # Gets a new action
            action = agent.act(state)

            # Takes action and calculates the total reward
            observation, reward, done, _ = env.step(action)
            total_reward += reward

            # Gets the next state
            next_state = np.reshape(observation, (1,) + env.observation_space.shape)

            # Memorizes the experience
            agent.memorize(state, action, reward, next_state, done)

            # Updates the online network weights
            if frame_count % agent.update_frequency == 0:
                agent.experience_reply()

            # Updates the target network weights
            if frame_count % agent.tau == 0:
                agent.update_target_network()

            # Updates the state
            state = next_state

            # Updates the total steps
            frame_count += 1

            if done:
                print("Episode %d/%d finished after %d episode steps with total reward = %f."
                      % (episode + 1, num_episodes, episode_step + 1, total_reward))
                break

            elif episode_step >= num_episode_steps - 1:
                print("Episode %d/%d timed out at %d with total reward = %f."
                      % (episode + 1, num_episodes, episode_step + 1, total_reward))

        # Updates the epsilon value
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Saves the online network weights
        if total_reward > max_reward:
            agent.save_weights("super_mario_bros_v0.h5")
            keras.backend.clear_session()

    # Closes the environment
    env.close()
