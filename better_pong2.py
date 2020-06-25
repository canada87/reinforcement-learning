from PIL import Image
import cv2
import numpy as np
import random
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import time

import keras.backend.tensorflow_backend as backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten


# ███████ ███████ ███    ██ ███████ ██ ██████  ██      ███████     ██████   █████  ██████   █████  ███    ███ ███████ ████████ ███████ ██████  ███████
# ██      ██      ████   ██ ██      ██ ██   ██ ██      ██          ██   ██ ██   ██ ██   ██ ██   ██ ████  ████ ██         ██    ██      ██   ██ ██
# ███████ █████   ██ ██  ██ ███████ ██ ██████  ██      █████       ██████  ███████ ██████  ███████ ██ ████ ██ █████      ██    █████   ██████  ███████
#      ██ ██      ██  ██ ██      ██ ██ ██   ██ ██      ██          ██      ██   ██ ██   ██ ██   ██ ██  ██  ██ ██         ██    ██      ██   ██      ██
# ███████ ███████ ██   ████ ███████ ██ ██████  ███████ ███████     ██      ██   ██ ██   ██ ██   ██ ██      ██ ███████    ██    ███████ ██   ██ ███████

#agent
# load_pre_trained_model = '4x8x3__140.00max__140.00avg__140.00min_0.98discount_0.99epsilondecay_1592587461.model'
load_pre_trained_model = None
replay_memory_size = 50_000 #dimensione massima di mosse che possono essere tenute a mente
min_replay_memory_size = 1000 #dimensione minima di mosse in memoria per poter cominciare a fare il training
sampling_memory = 100 #numenro di mosse usate per il training (sotto campione di replay_memory)
discount = 0.98

#game
go_live = True
episodes = 50 #numero massimo di partite
sampling_epoc = 10 #ogni quante epoche registrare i risultati
how_aften_go_live = 25 #ogni quante epoche fare il live di cosa succede
min_reward_bar = -70
model_name = '3x3'

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
# epsilon_decay = 0.999#cosi dopo 100 eppisodi epsilon scende al 0.9
epsilon_decay = 0.999
min_epsilon = 0.001


#  ██████   ██████   ██████  ███████ ████████ ████████ ██
# ██    ██ ██       ██       ██         ██       ██    ██
# ██    ██ ██   ███ ██   ███ █████      ██       ██    ██
# ██    ██ ██    ██ ██    ██ ██         ██       ██    ██
#  ██████   ██████   ██████  ███████    ██       ██    ██

class element:
    def __init__(self, x, y, x_speed, y_speed, x_size, y_size, w_size, color):
        self.x = x
        self.y = y
        self.x_speed = x_speed
        self.y_speed = y_speed
        self.x_size = x_size
        self.y_size = y_size
        self.w_size = w_size
        self.color = color

    def display(self, background):
        ball_pixels = np.zeros((self.w_size, self.w_size, 3), dtype=np.uint8)
        ball_pixels[self.x - self.x_size : self.x + self.x_size, self.y - self.y_size : self.y + self.y_size] = self.color
        background +=ball_pixels

    def action(self, choice):
        if choice == 0:
            self.move(y=1)
        elif choice == 1:
            self.move(y=-1)
        elif choice == 2:
            self.move(y=0)

    def move(self, y):
        self.y += y
        if self.y > self.w_size - self.y_size:
            self.y = self.w_size - self.y_size
        if self.y < self.y_size:
            self.y = self.y_size

    def update(self):
        self.x += self.x_speed
        self.y += self.y_speed

    def win_lose(self):
        lose_side = 0
        if self.x > self.w_size - self.x_size:
            lose_side = 1
            return True, lose_side
        if self.x < self.x_size:
            lose_side = 2
            return True, lose_side
        return False, lose_side

    def boundaries(self):
        if self.y + self.y_size> self.w_size:
            self.y = self.w_size - self.y_size
            self.y_speed *= -1

        if self.y < self.y_size:
            self.y = self.y_size
            self.y_speed *= -1


    def hit(self, other):
        is_hit = 0
        if (self.y < other.y + other.y_size + self.y_size and self.y > other.y - other.y_size - self.y_size ) and (self.x < other.x + other.x_size + self.x_size and self.x > other.x - other.x_size - self.x_size):
            if self.x < self.w_size//2:
                if self.x < other.x + other.x_size:
                    self.x_speed = self.x_speed
                else:
                    self.x_speed *= -1
                    is_hit = 1

            if self.x > self.w_size//2:
                if self.x > other.x - other.x_size:
                    self.x_speed = self.x_speed
                else:
                    self.x_speed *= -1
                    is_hit = 2
        return is_hit


# ███████ ███    ██ ██    ██ ██ ██████   ██████  ███    ██ ███    ███ ███████ ███    ██ ████████
# ██      ████   ██ ██    ██ ██ ██   ██ ██    ██ ████   ██ ████  ████ ██      ████   ██    ██
# █████   ██ ██  ██ ██    ██ ██ ██████  ██    ██ ██ ██  ██ ██ ████ ██ █████   ██ ██  ██    ██
# ██      ██  ██ ██  ██  ██  ██ ██   ██ ██    ██ ██  ██ ██ ██  ██  ██ ██      ██  ██ ██    ██
# ███████ ██   ████   ████   ██ ██   ██  ██████  ██   ████ ██      ██ ███████ ██   ████    ██


class env:
    size = 500
    ball_size = 10
    x_speed = 4
    y_speed = 6

    reward_lose_game = -100
    reward_win_game = 100
    reward_hit = 10
    reward_still_alive = 0

    def set_up(self):
        self.ball = element(self.size//2, self.size//2, self.x_speed, self.y_speed, self.ball_size, self.ball_size, self.size, (255, 175, 0))
        self.pad1 = element(25, self.size//4, 0, 0, 10, 50, self.size, (0, 255, 0))
        self.pad2 = element(self.size-25, 350, 0, 0, 10, 50, self.size, (0, 0, 255))

        observation = self.ball.y, self.pad1.x, self.pad2.y
        return np.array(observation)

    def step(self, action):
        self.pad1.action(action)

        self.pad2.y = self.ball.y

        self.pad1.boundaries()
        self.pad2.boundaries()
        self.ball.boundaries()

        is_hit = self.ball.hit(self.pad1)
        self.ball.hit(self.pad2)

        self.ball.update()

        done, lose_side = self.ball.win_lose()
        observation = self.ball.y, self.pad1.x, self.pad2.y

        reward = self.reward_still_alive

        if is_hit == 1:
            reward = self.reward_hit

        if done:
            if lose_side == 2:
                reward = self.reward_lose_game
            if lose_side == 1:
                reward = self.reward_win_game

        return np.array(observation), reward, done

    def render(self):
        screen = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        self.ball.display(screen)
        self.pad1.display(screen)
        self.pad2.display(screen)

        img = Image.fromarray(screen, 'RGB')
        cv2.imshow("image", np.array(img))
        cv2.waitKey(20)


        #  █████   ██████  ███████ ███    ██ ████████
        # ██   ██ ██       ██      ████   ██    ██
        # ███████ ██   ███ █████   ██ ██  ██    ██
        # ██   ██ ██    ██ ██      ██  ██ ██    ██
        # ██   ██  ██████  ███████ ██   ████    ██

class pong_agent:
    def __init__(self, action_space, observation_space, replay_memory_size, epsilon, min_epsilon, epsilon_decay):
        self.action_space = 3
        self.observation_space = 3
        self.model = self.create_model()
        self.replay_memory = deque(maxlen = replay_memory_size)

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

    def create_model(self):
        if load_pre_trained_model is not None:
            print('load {}'.format(load_pre_trained_model))
            model = load_model('models/'+load_pre_trained_model)
            print('model {} loaded'.format(load_pre_trained_model))
        else:
            model = Sequential()
            model.add(Dense(self.observation_space, activation = 'relu', input_dim =  self.observation_space))
            # model.add(Dense(64, activation = 'relu'))
            # model.add(Dropout(0.2))
            model.add(Dense(self.action_space, activation = 'linear'))


            # model.add(Flatten(input_shape =  (1,) + self.observation_space.shape))
            # model.add(Dense(self.action_space, activation = 'softmax'))


            model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model

    def update_replay_memory(self, frame):
        self.replay_memory.append(frame)

    def train(self):
        if len(self.replay_memory) < min_replay_memory_size:
            return
        minibatch = random.sample(self.replay_memory, sampling_memory)

        observation = np.array([frame[0] for frame in minibatch])
        action_predict = self.model.predict(observation)
        next_observation = np.array([frame[3] for frame in minibatch])
        next_action_predict = self.model.predict(next_observation)

        x = []
        y = []

        for index, (observation, action, reward, next_observation, done) in enumerate(minibatch):
            if not done:
                best_next_action = np.max(next_action_predict[index])
                next_action = reward + discount * best_next_action
            else:
                next_action = reward

            current_predicted_action = action_predict[index]
            current_predicted_action[action] = next_action

            x.append(observation)
            y.append(current_predicted_action)

        # self.model.fit(np.array(x), np.array(y), batch_size = sampling_memory, verbose = 0, shuffle=False)
        self.model.fit(np.array(x), np.array(y), verbose = 0, shuffle=False)

    def decide_the_next_action(self, observation):

        if np.random.random()>self.epsilon:
            action = np.argmax(self.model.predict(np.array(observation).reshape(-1, *observation.shape))[0])
        else:
            action = np.random.randint(0, self.action_space)

        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)

        return action
        # return  np.argmax(self.model.predict(np.array(observation).reshape(-1, *observation.shape))[0])


# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████

ep_rewards = []
aggr_ep_rewards = {'ep' : [], 'avg': [], 'min': [], 'max': []}

agent = pong_agent(action_space = 3, observation_space = 3, replay_memory_size = replay_memory_size, epsilon = epsilon, min_epsilon = min_epsilon, epsilon_decay = epsilon_decay)
environment = env()
for episode in tqdm(range(1, episodes+1), ascii = True, unit='episode'):
    episode_reward = 0

    observation = environment.set_up()
    done = False

###############################################################################################
    while not done:

        action = agent.decide_the_next_action(observation)

        next_observation, reward, done = environment.step(action)
        episode_reward += reward

        agent.update_replay_memory((observation, action, reward, next_observation, done))
        observation = next_observation

        if go_live and not episode % how_aften_go_live:
            environment.render()
###############################################################################################
    agent.train()

    ep_rewards.append(episode_reward)
    if not episode % sampling_epoc or episode == 1:
        average_reward = sum(ep_rewards[-sampling_epoc:])/len(ep_rewards[-sampling_epoc:])
        min_reward = min(ep_rewards[-sampling_epoc:])
        max_reward = max(ep_rewards[-sampling_epoc:])

        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min_reward)
        aggr_ep_rewards['max'].append(max_reward)

        # Save model, but only when min reward is greater or equal a set value
        if average_reward >= min_reward_bar:
            agent.model.save(f'models/{model_name}_{average_reward:_>7.2f}avg_{discount}discount_{epsilon_decay}epsilondecay_{int(time.time())}.model')
            # agent.model.save(f'models/{model_name}_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min_{discount}discount_{epsilon_decay}epsilondecay_{int(time.time())}.model')



plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label = 'avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label = 'min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label = 'max')
plt.legend(loc = 4)
plt.show()
