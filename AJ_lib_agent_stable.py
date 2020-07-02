import numpy as np
import random
from collections import deque
import time

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam


# ██████   ██████  ███    ██
# ██   ██ ██    ██ ████   ██
# ██   ██ ██    ██ ██ ██  ██
# ██   ██ ██ ▄▄ ██ ██  ██ ██
# ██████   ██████  ██   ████
#             ▀▀

class Agent_DQN:
    def __init__(self, action_space, observation_space, replay_memory_size, epsilon, min_epsilon, epsilon_decay, discount,
                 sampling_memory, load_pre_trained_model):
        '''
        :param action_space: number of actions can be taken
        :param observation_space: number of observations
        :param replay_memory_size: max len of the frames kept in mind
        :param epsilon: starting value for the exploration
        :param min_epsilon: minimum value for the exploration
        :param epsilon_decay: rate of decay for the exploration
        :param discount: discount for the future actions
        :param sampling_memory: how many frame used for the training
        :param load_pre_trained_model: name of the model to load if present
        '''

        # change the type of action selection in the ddqn function (False is more reliable)
        self.second_type_of_brain = False

        self.load_pre_trained_model = load_pre_trained_model

        self.action_space = action_space
        self.observation_space = observation_space

        # initialize the networks
        self.live_model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.live_model.get_weights())
        self.learning_rate = 0.001

        self.replay_memory = deque(maxlen = replay_memory_size)

        self.discount = discount
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.sampling_memory = sampling_memory

        self.motion_recognition = False # use small random batches of sequential frames instead of totaly random
        self.sequential_frame = 4 # number of sequential frame saved, used to give the idea of motion
        self.sequential_minibatch = deque(maxlen = self.sampling_memory) # memory used to train the network if using the small batches

        #save the performances over time
        self.ep_rewards = []
        self.aggr_ep_rewards = {'ep' : [], 'avg': [], 'min': [], 'max': []}

    def create_model(self):
        '''
        create the new model or load an old one
        :return: model
        '''
        if self.load_pre_trained_model is not None:
            print('load {}'.format(self.load_pre_trained_model))
            model = load_model('models/'+self.load_pre_trained_model)
            print('model {} loaded'.format(self.load_pre_trained_model))
        else:
            model = Sequential()
            model.add(Dense(64, activation = 'linear', input_dim =  self.observation_space))
            model.add(Dense(self.action_space, activation = 'linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def update_replay_memory(self, frame):
        '''
        save each frame in memory
        :param frame: (observation, action, reward, next_action, done)
        :param observation: numpy array of floats or ints
        :param action: int
        :param reward: int
        :param next_action: numpy array of floats or ints
        :param done: boolean
        '''
        self.replay_memory.append(frame)

    def create_batches(self):
        '''
        creates small batches if sequential frames to keep the information of the motion
        if used it slow down the training speed
        :return: deque list semi-ordered with a maxlen of sampling_memory
        '''
        stack_len = self.sequential_frame #number of sequential frames stacked in a single batch
        batch_len = self.sampling_memory//stack_len#number of batches

        #index to select random batches
        rand_index = random.sample([i for i in range(len(self.replay_memory) - stack_len)], batch_len)

        #load the memory with the batches
        for i in rand_index:
            for j in range(stack_len):
                self.sequential_minibatch.append(self.replay_memory[i+j])
        return self.sequential_minibatch

    def DDQN_train(self, episode, how_aften_train, how_aften_replace_target):
        '''
        Doubel Deep Q learning
        :param episode: number of the match is going on
        :param how_aften_train: how aften train the live model
        :param how_aften_replace_target: how aften update the target model
        '''
        # the training is performed only if the memory is larger then the sampling memory and every how_aften_train episodes
        if len(self.replay_memory) > self.sampling_memory and episode % how_aften_train == 0:
            if self.motion_recognition:
                # the minibatch is created with random batches of sequential frames
                minibatch = self.create_batches()
            else:
                # the minibatch is created with random frames
                minibatch = random.sample(self.replay_memory, self.sampling_memory)

            # extract the observation from minibatch and predict the action with live model
            observation = np.array([frame[0] for frame in minibatch])
            action_predict = self.live_model.predict(observation)

            # extract the next_observation from minibatch and predict the next_action with target model
            next_observation = np.array([frame[3] for frame in minibatch])
            next_action_predict = self.target_model.predict(next_observation)

            if self.second_type_of_brain:
                #use the live model to predict the next action with an hybrin solution
                next_action_predict_live = self.live_model.predict(next_observation)

            x = []
            y = []

            for index, (observation, action, reward, next_observation, done) in enumerate(minibatch):
                if self.second_type_of_brain: #questo e' il modello del tipo che ci piace
                    best_next_action_index = np.argmax(next_action_predict_live[index])
                    best_next_action = next_action_predict[index, best_next_action_index]
                    next_action = reward + (self.discount * best_next_action)*(1-done)
                else: #questo e' il modello del tipo che non ci piace
                    #select the most likely action with the target model (as the one with the larger number)
                    best_next_action = np.max(next_action_predict[index])
                    #use the number to generate the next value for the prediction table
                    next_action = reward + (self.discount * best_next_action)*(1-done)

                # upload the new number on the action/selection table
                current_predicted_action = action_predict[index]
                current_predicted_action[action] = next_action

                x.append(observation)
                y.append(current_predicted_action)

            #fit the live model
            self.live_model.fit(np.array(x), np.array(y), verbose = 0, shuffle=False)

            if episode % how_aften_replace_target == 0:
                # upload the target model
                self.target_model.set_weights(self.live_model.get_weights())

    def DQN_train(self, episode, how_aften_train):
        '''
        Deep Q learning
        :param episode: number of the match is going on
        :param how_aften_train: how aften train the live model
        '''
        # the training is performed only if the memory is larger then the sampling memory and every how_aften_train episodes
        if len(self.replay_memory) > self.sampling_memory and episode % how_aften_train == 0:
            if self.motion_recognition:
                # the minibatch is created with random batches of sequential frames
                minibatch = self.create_batches()
            else:
                # the minibatch is created with random frames
                minibatch = random.sample(self.replay_memory, self.sampling_memory)

            # extract the observation from minibatch and predict the action with live model
            observation = np.array([frame[0] for frame in minibatch])
            action_predict = self.live_model.predict(observation)

            # extract the next_observation from minibatch and predict the next_action with live model
            next_observation = np.array([frame[3] for frame in minibatch])
            next_action_predict = self.live_model.predict(next_observation)

            x = []
            y = []

            for index, (observation, action, reward, next_observation, done) in enumerate(minibatch):
                #select the most likely action (as the one with the larger number)
                best_next_action = np.max(next_action_predict[index])
                #use the number to generate the next value for the prediction table
                next_action = reward + (self.discount * best_next_action)*(1-done)

                # upload the new number on the action/selection table
                current_predicted_action = action_predict[index]
                current_predicted_action[action] = next_action

                x.append(observation)
                y.append(current_predicted_action)

            #fit the live model
            self.live_model.fit(np.array(x), np.array(y), verbose = 0, shuffle=False)

    def decide_the_next_action(self, observation):
        '''
        the model uses the observation to predict the next action
        :param observation: numpy array of floats or ints
        :return: action (int)
        '''
        if np.random.random()>self.epsilon:
            # predicted action
            action = np.argmax(self.live_model.predict(np.array(observation).reshape(-1, *observation.shape))[0])
        else:
            # randomly a random action is selected
            action = np.random.randint(0, self.action_space)
        return action

    def update_epsilon(self):
        '''
        update the decay of epslion to reduce the exploration and increas the exploitation
        '''
        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)

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
                self.live_model.save(f'models/{model_name}_{episode}episode_{max_reward}max_{average_reward}avg_{min_reward}min_{self.discount}discount_{self.epsilon_decay}epsilondecay_{int(time.time())}.model')



# ██████   ██████  ██      ██  ██████ ██    ██      ██████  ██████   █████  ██████  ██ ███████ ███    ██ ████████
# ██   ██ ██    ██ ██      ██ ██       ██  ██      ██       ██   ██ ██   ██ ██   ██ ██ ██      ████   ██    ██
# ██████  ██    ██ ██      ██ ██        ████       ██   ███ ██████  ███████ ██   ██ ██ █████   ██ ██  ██    ██
# ██      ██    ██ ██      ██ ██         ██        ██    ██ ██   ██ ██   ██ ██   ██ ██ ██      ██  ██ ██    ██
# ██       ██████  ███████ ██  ██████    ██         ██████  ██   ██ ██   ██ ██████  ██ ███████ ██   ████    ██


import keras.backend as K
from keras.layers import Input
from keras.models import Model

class Agent_policy_gradient:
    def __init__(self, action_space, observation_space, epsilon_decay, load_pre_trained_model):
        '''
        :param action_space: number of actions can be taken
        :param observation_space: number of observations
        :param epsilon_decay: rate of decay in the G calculation
        :param name_model: name of the model to load if present
        '''

        self.load_pre_trained_model = load_pre_trained_model

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

        input = Input(shape = (self.observation_space,))
        advantages = Input(shape = [1])
        dense1 = Dense(64, activation='relu')(input)
        probs = Dense(self.action_space, activation='softmax')(dense1)

        policy = Model(input = [input, advantages], output = [probs])
        policy.compile(optimizer = Adam(lr = self.learning_rate), loss = custom_loss)

        predict = Model(input = [input], output = [probs])
        # policy.summary()
        # predict.summary()
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

    def update_replay_memory(self, frame):
        '''
        save each frame in memory
        :param frame: (observation, action, reward)
        :param observation: numpy array of floats or ints
        :param action: int
        :param reward: int
        '''
        self.observation.append(frame[0])
        self.action.append(frame[1])
        self.reward.append(frame[2])

    def policy_gradient_learning(self):
        '''
        learning step
        have to be placed outside the match loop
        it learn with the memory of an entire match and then it clear the memory
        :return: cost
        '''
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
                self.policy.save(f'models/policy_{model_name}_{episode}episode_{max_reward}max_{average_reward}avg_{min_reward}min_{self.epsilon_decay}epsilondecay_{int(time.time())}.model')
                self.predict.save(f'models/predict_{model_name}_{episode}episode_{max_reward}max_{average_reward}avg_{min_reward}min_{self.epsilon_decay}epsilondecay_{int(time.time())}.model')
