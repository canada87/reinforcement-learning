import numpy as np
import random
from collections import deque
import time


from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras import initializers


#  █████   ██████  ███████ ███    ██ ████████
# ██   ██ ██       ██      ████   ██    ██
# ███████ ██   ███ █████   ██ ██  ██    ██
# ██   ██ ██    ██ ██      ██  ██ ██    ██
# ██   ██  ██████  ███████ ██   ████    ██

class Agent:
    def __init__(self, action_space, observation_space, replay_memory_size, epsilon, min_epsilon, epsilon_decay, discount, sampling_memory, load_pre_trained_model):

        self.load_pre_trained_model = load_pre_trained_model

        self.action_space = action_space
        self.observation_space = observation_space

        self.live_model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.live_model.get_weights())

        self.replay_memory = deque(maxlen = replay_memory_size)

        self.discount = discount
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.sampling_memory = sampling_memory

        self.ep_rewards = []
        self.aggr_ep_rewards = {'ep' : [], 'avg': [], 'min': [], 'max': []}

    def create_model(self):
        if self.load_pre_trained_model is not None:
            print('load {}'.format(self.load_pre_trained_model))
            model = load_model('models/'+self.load_pre_trained_model)
            print('model {} loaded'.format(self.load_pre_trained_model))
        else:
            model = Sequential()
            model.add(Dense(self.observation_space, activation = 'sigmoid', input_dim =  self.observation_space))
            model.add(Dense(self.action_space, activation = 'linear', bias_initializer=initializers.Zeros()))
            model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, frame):
        self.replay_memory.append(frame)

    def DDQN_train(self, episode, how_aften_train, how_aften_replace_target):
        if len(self.replay_memory) > self.sampling_memory and episode % how_aften_train == 0:
            minibatch = random.sample(self.replay_memory, self.sampling_memory)

            observation = np.array([frame[0] for frame in minibatch])
            action_predict = self.live_model.predict(observation)

            next_observation = np.array([frame[3] for frame in minibatch])
            # next_action_predict_live = self.live_model.predict(next_observation)
            next_action_predict = self.target_model.predict(next_observation)

            x = []
            y = []

            for index, (observation, action, reward, next_observation, done) in enumerate(minibatch):
                best_next_action = np.max(next_action_predict[index])
                next_action = reward + (self.discount * best_next_action)*(1-done)

                # best_next_action_index = np.argmax(next_action_predict_live[index])
                # best_next_action = next_action_predict[index, best_next_action_index]
                # next_action = reward + (self.discount * best_next_action)(1-done)

                current_predicted_action = action_predict[index]
                current_predicted_action[action] = next_action

                x.append(observation)
                y.append(current_predicted_action)

            self.live_model.fit(np.array(x), np.array(y), verbose = 0, shuffle=False)

            if episode % how_aften_replace_target == 0:
                self.target_model.set_weights(self.live_model.get_weights())

    def DQN_train(self, episode, how_aften_train):
        if len(self.replay_memory) > self.sampling_memory and episode % how_aften_train == 0:
            minibatch = random.sample(self.replay_memory, self.sampling_memory)

            observation = np.array([frame[0] for frame in minibatch])
            action_predict = self.live_model.predict(observation)

            next_observation = np.array([frame[3] for frame in minibatch])
            next_action_predict = self.live_model.predict(next_observation)

            x = []
            y = []

            for index, (observation, action, reward, next_observation, done) in enumerate(minibatch):
                best_next_action = np.max(next_action_predict[index])
                next_action = reward + (self.discount * best_next_action)*(1-done)

                current_predicted_action = action_predict[index]
                current_predicted_action[action] = next_action

                x.append(observation)
                y.append(current_predicted_action)

            self.live_model.fit(np.array(x), np.array(y), verbose = 0, shuffle=False)

    def decide_the_next_action(self, observation):
        if np.random.random()>self.epsilon:
            action = np.argmax(self.live_model.predict(np.array(observation).reshape(-1, *observation.shape))[0])
        else:
            action = np.random.randint(0, self.action_space)

        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)

        return action

    def save_model(self, episode_reward, episode, episodes, sampling_epoc, min_reward_bar, model_name, save_enable = True):
        self.ep_rewards.append(episode_reward)

        if not episode % sampling_epoc or episode == 1 or episode == episodes - 1:
            average_reward = sum(self.ep_rewards[-sampling_epoc:])/len(self.ep_rewards[-sampling_epoc:])
            min_reward = min(self.ep_rewards[-sampling_epoc:])
            max_reward = max(self.ep_rewards[-sampling_epoc:])

            self.aggr_ep_rewards['ep'].append(episode)
            self.aggr_ep_rewards['avg'].append(average_reward)
            self.aggr_ep_rewards['min'].append(min_reward)
            self.aggr_ep_rewards['max'].append(max_reward)

            # Save model, but only when min reward is greater or equal a set value
            if save_enable and (average_reward >= min_reward_bar or episode == episodes - 1):
                self.live_model.save(f'models/{model_name}_{episode}episode_{max_reward}max_{average_reward}avg_{min_reward}min_{self.discount}discount_{self.epsilon_decay}epsilondecay_{int(time.time())}.model')
