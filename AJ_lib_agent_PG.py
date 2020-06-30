import numpy as np

import time

from keras.models import load_model, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import keras.backend as K

class Agent_policy_gradient:
    def __init__(self, action_space, observation_space, epsilon_decay, load_pre_trained_policy, load_pre_trained_predict):
        '''
        :param action_space: number of actions can be taken
        :param observation_space: number of observations
        :param epsilon_decay: rate of decay in the G calculation
        :param name_model: name of the model to load if present
        '''

        self.load_pre_trained_policy = load_pre_trained_policy
        self.load_pre_trained_predict = load_pre_trained_predict

        self.action_space = action_space
        self.observation_space = observation_space

        # initialize the networks
        self.learning_rate = 0.001
        self.policy, self.predict = self.create_policy_network_model()

        self.observation = []
        self.action = []
        self.reward = []

        self.G = 0
        self.epsilon_decay = epsilon_decay

        #save the performances over time
        self.ep_rewards = []
        self.aggr_ep_rewards = {'ep' : [], 'avg': [], 'min': [], 'max': []}

    def create_policy_network_model(self):
        '''
        create the new model or load an old one
        :return: policy, predict (2 different models)
        '''
        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)
            return K.sum(-log_lik*advantages)

        if self.load_pre_trained_policy is not None:
            print('load {}'.format(self.load_pre_trained_policy))
            policy = load_model('models/'+self.load_pre_trained_policy)
            print('model {} loaded'.format(self.load_pre_trained_policy))
            print('load {}'.format(self.load_pre_trained_predict))
            predict = load_model('models/'+self.load_pre_trained_predict)
            print('model {} loaded'.format(self.load_pre_trained_predict))
        else:
            input = Input(shape = (self.observation_space,))
            advantages = Input(shape = [1])
            dense1 = Dense(64, activation='relu')(input)
            probs = Dense(self.action_space, activation='softmax')(dense1)

            policy = Model(input = [input, advantages], output = [probs])
            policy.compile(optimizer = Adam(lr = self.learning_rate), loss = custom_loss)

            predict = Model(input = [input], output = [probs])
        return policy, predict

    def decide_the_next_action(self, observation):
        '''
        the model uses the observation to predict the next action
        :param observation: numpy array of floats or ints
        :return: action (int)
        '''

        #reshape in order to be used in the prediction
        observation = observation[np.newaxis, :]
        #predict the probability for each possible action
        probabilities = self.predict.predict(observation)[0]
        #choose a random action accordingly with the probability disctribution just predicted
        action = np.random.choice(self.action_space, p = probabilities)
        return action

    def update_memory(self, observation, action, reward):
        '''
        save each frame in memory
        :param frame: observation, action, reward
        :param observation: numpy array of floats or ints
        :param action: int
        :param reward: int
        '''
        self.observation.append(observation)
        self.action.append(action)
        self.reward.append(reward)

    def policy_gradient_learning(self):
        observation = np.array(self.observation)
        action = np.array(self.action)
        reward = np.array(self.reward)

        #create a OneHotEncoder of the action vector
        actions = np.zeros([len(action), self.action_space])
        actions[np.arange(len(action)), action] = 1

        G = np.zeros_like(reward)
        for t in range(len(reward)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward)):
                G_sum += reward[k]*discount
                discount *= self.epsilon_decay
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G - mean)/std

        cost = self.policy.train_on_batch([observation, self.G], actions)

        self.observation = []
        self.action = []
        self.reward = []

        return cost

    def save_model(self, episode_reward, episode, episodes, sampling_epoc, min_reward_bar, model_name, save_enable = True):
        '''
        save the model and the performances
        :param episode_reward: reward obtained in a full match
        :param episode: current episode
        :param episodes: max number of episodes
        :param sampling_epoc: average the performances over that period of epocs
        :param min_reward_bar: min reward requested to save the model
        :param model_name: model name
        :param save_enable: active the save model
        '''

        #update the rewards hystory
        self.ep_rewards.append(episode_reward)

        if not episode % sampling_epoc or episode == 1 or episode == episodes:
            average_reward = sum(self.ep_rewards[-sampling_epoc:])/len(self.ep_rewards[-sampling_epoc:])
            min_reward = min(self.ep_rewards[-sampling_epoc:])
            max_reward = max(self.ep_rewards[-sampling_epoc:])

            self.aggr_ep_rewards['ep'].append(episode)
            self.aggr_ep_rewards['avg'].append(average_reward)
            self.aggr_ep_rewards['min'].append(min_reward)
            self.aggr_ep_rewards['max'].append(max_reward)

            # Save model, but only when min reward is greater or equal a set value
            if save_enable and (average_reward >= min_reward_bar or episode == episodes):
                self.policy.save(f'models/policy_{model_name}_{episode}episode_{max_reward}max_{average_reward}avg_{min_reward}min_{self.discount}discount_{self.epsilon_decay}epsilondecay_{int(time.time())}.model')
                self.predict.save(f'models/predict_{model_name}_{episode}episode_{max_reward}max_{average_reward}avg_{min_reward}min_{self.discount}discount_{self.epsilon_decay}epsilondecay_{int(time.time())}.model')
