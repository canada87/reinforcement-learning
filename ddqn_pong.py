from PIL import Image
import cv2
import numpy as np
import random
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import time


from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras import initializers



# ███████ ███████ ███    ██ ███████ ██ ██████  ██      ███████     ██████   █████  ██████   █████  ███    ███ ███████ ████████ ███████ ██████  ███████
# ██      ██      ████   ██ ██      ██ ██   ██ ██      ██          ██   ██ ██   ██ ██   ██ ██   ██ ████  ████ ██         ██    ██      ██   ██ ██
# ███████ █████   ██ ██  ██ ███████ ██ ██████  ██      █████       ██████  ███████ ██████  ███████ ██ ████ ██ █████      ██    █████   ██████  ███████
#      ██ ██      ██  ██ ██      ██ ██ ██   ██ ██      ██          ██      ██   ██ ██   ██ ██   ██ ██  ██  ██ ██         ██    ██      ██   ██      ██
# ███████ ███████ ██   ████ ███████ ██ ██████  ███████ ███████     ██      ██   ██ ██   ██ ██   ██ ██      ██ ███████    ██    ███████ ██   ██ ███████

#agent
load_pre_trained_model = '3x3_999episode__618.00max__513.10avg__105.00min_0.99discount_0.9999epsilondecay_1593110440.model'
# load_pre_trained_model = None
replay_memory_size = 50_000 #dimensione massima di mosse che possono essere tenute a mente
sampling_memory = 64 #numenro di mosse usate per il training (sotto campione di replay_memory)
discount = 0.99

#game
go_live = True
episodes = 1000 #numero massimo di partite
sampling_epoc = 10 #ogni quante epoche registrare i risultati
how_aften_go_live = 500 #ogni quante epoche fare il live di cosa succede
how_aften_replace_target = 100 #ogni quanto caricare i pesi da modello live a modello target
how_aften_train = 4
min_reward_bar = 1500#valore minimo di reward per salvare i pesi del modello
model_name = '3x3'

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
epsilon_decay = 0.99999
min_epsilon = 0.1


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

        if (np.abs(self.y - other.y) < other.y_size + self.y_size):
            if (np.abs(other.x + other.x_size - self.x) < self.x_size):
                if self.x - self.x_size < other.x + other.x_size - 3:
                    self.x_speed = self.x_speed
                else:
                    self.x_speed *= -1
                    is_hit = 1

        if (np.abs(self.y - other.y) < other.y_size + self.y_size):
            if (np.abs(other.x - other.x_size - self.x) < self.x_size):
                if self.x + self.x_size > other.x - other.x_size + 3:
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

    def __init__(self):
        self.size = 500
        self.ball_size = 10
        self.x_speed = 2
        self.y_speed = 3

        self.reward_lose_game = -300
        self.reward_win_game = 300
        self.reward_hit = 30
        self.reward_still_alive = 1
        self.reward_ball_on_eye = 2

    def set_up(self):
        self.ball = element(self.size//2, self.size//2, self.x_speed, self.y_speed, self.ball_size, self.ball_size, self.size, (255, 175, 0))
        self.pad1 = element(25, self.size//2, 0, 0, 10, 50, self.size, (0, 255, 0))
        self.pad2 = element(self.size-25, self.size//2, 0, 0, 10, 50, self.size, (0, 0, 255))

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

        if self.ball.y < self.pad1.y + self.pad1.y_size and self.ball.y > self.pad1.y - self.pad1.y_size:
            reward = self.reward_ball_on_eye

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
        cv2.waitKey(10)


#  █████   ██████  ███████ ███    ██ ████████
# ██   ██ ██       ██      ████   ██    ██
# ███████ ██   ███ █████   ██ ██  ██    ██
# ██   ██ ██    ██ ██      ██  ██ ██    ██
# ██   ██  ██████  ███████ ██   ████    ██

class pong_agent:
    def __init__(self, action_space, observation_space, replay_memory_size, epsilon, min_epsilon, epsilon_decay, discount, sampling_memory):
        self.action_space = 3
        self.observation_space = 3

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
        if load_pre_trained_model is not None:
            print('load {}'.format(load_pre_trained_model))
            model = load_model('models/'+load_pre_trained_model)
            print('model {} loaded'.format(load_pre_trained_model))
        else:
            model = Sequential()
            # model.add(Dense(self.observation_space, activation = 'relu', input_dim =  self.observation_space))
            # # model.add(Dense(64, activation = 'relu'))
            # # model.add(Dropout(0.2))
            # model.add(Dense(self.action_space, activation = 'linear'))

            # model.add(Dense(self.observation_space, activation = 'relu', input_dim =  self.observation_space))
            # model.add(Dense(self.action_space, activation = 'softmax'))
            # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

            model.add(Dense(self.observation_space, activation = 'sigmoid', input_dim =  self.observation_space))
            model.add(Dense(self.action_space, activation = 'linear', bias_initializer=initializers.Zeros()))
            model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model

    def update_replay_memory(self, frame):
        self.replay_memory.append(frame)

    def train(self, episode, how_aften_train, how_aften_replace_target):
        if len(self.replay_memory) > self.sampling_memory and episode % how_aften_train == 0:

            # minibatch = random.sample(self.replay_memory, self.sampling_memory)
            #
            # observation = np.array([frame[0] for frame in minibatch])
            # action = np.array([frame[1] for frame in minibatch])
            # reward = np.array([frame[2] for frame in minibatch])
            # next_observation = np.array([frame[3] for frame in minibatch])
            # done = np.array([frame[4] for frame in minibatch])
            #
            # action_predict = self.live_model.predict(observation)
            # next_action_predict_live = self.live_model.predict(next_observation)
            # next_action_predict = self.target_model.predict(next_observation)
            #
            # max_actions = np.argmax(next_action_predict_live, axis = 1)
            # sample_index = np.arange(self.sampling_memory, dtype=np.int32)
            #
            # # print()
            # # print(action_predict[0])
            # # print('azione presa', action[0])
            # # print('azione predetta per il prossimo turno', max_actions.astype(int)[0])
            #
            # action_predict[sample_index, action] = reward + self.discount*next_action_predict[sample_index, max_actions.astype(int)]*(1-done)
            #
            # # print()
            # # print(self.discount*next_action_predict[0])
            # # print(reward[0])
            # # print(action_predict[0])
            # # print()
            #
            # self.live_model.fit(observation, action_predict, verbose = 0, shuffle=False)
            #
            # if episode % how_aften_replace_target == 0:
            #     self.target_model.set_weights(self.live_model.get_weights())

##############################################################################################################

            # minibatch = random.sample(self.replay_memory, self.sampling_memory)
            #
            # observation = np.array([frame[0] for frame in minibatch])
            # action = np.array([frame[1] for frame in minibatch])
            # reward = np.array([frame[2] for frame in minibatch])
            # next_observation = np.array([frame[3] for frame in minibatch])
            # done = np.array([frame[4] for frame in minibatch])
            #
            # next_action_predict = self.target_model.predict(next_observation)
            #
            # sample_index = np.arange(self.sampling_memory, dtype=np.int32)
            # action_predict = reward + self.discount*np.max(next_action_predict[sample_index], axis = 1)*(1-done)
            # print()
            # print(action_predict[0])
            # self.live_model.fit(observation, action_predict, verbose = 0, shuffle=False)
            #
            #
            # print('azione presa', action[0])
            # print('azione predetta per il prossimo turno', max_actions.astype(int)[0])
            # print()
            # print(self.discount*next_action_predict[0])
            # print(reward[0])
            # print(action_predict[0])
            # print()
            #
            # if episode % how_aften_replace_target == 0:
            #     self.target_model.set_weights(self.live_model.get_weights())

##############################################################################################################

            minibatch = random.sample(self.replay_memory, self.sampling_memory)

            observation = np.array([frame[0] for frame in minibatch])
            action_predict = self.live_model.predict(observation)

            next_observation = np.array([frame[3] for frame in minibatch])
            # next_action_predict_live = self.live_model.predict(next_observation)
            next_action_predict = self.target_model.predict(next_observation)

            x = []
            y = []

            for index, (observation, action, reward, next_observation, done) in enumerate(minibatch):
                if not done:
                    best_next_action = np.max(next_action_predict[index])
                    next_action = reward + self.discount * best_next_action

                    # best_next_action_index = np.argmax(next_action_predict_live[index])
                    # best_next_action = next_action_predict[index, best_next_action_index]
                    # next_action = reward + self.discount * best_next_action

                    # print()
                    # print('best action live', next_action_predict_live[index]
                    # print('best action live index', best_next_action_index)
                    # print('best action', np.argmax(next_action_predict[index]))
                    # print('next_action_predict', next_action_predict[index])
                    # print('best_next_action', best_next_action)
                    # print('reward', reward)
                    # print('next_action', next_action)
                    # print('action_predict', action_predict[index])

                else:
                    next_action = reward

                current_predicted_action = action_predict[index]
                current_predicted_action[action] = next_action

                # print('action',action)
                # print('current_predicted_action',current_predicted_action)
                # print()

                x.append(observation)
                y.append(current_predicted_action)

            # self.live_model.fit(np.array(x), np.array(y), batch_size = sampling_memory, verbose = 0, shuffle=False)
            self.live_model.fit(np.array(x), np.array(y), verbose = 0, shuffle=False)

            if episode % how_aften_replace_target == 0:
                self.target_model.set_weights(self.live_model.get_weights())


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
        # return  np.argmax(self.live_model.predict(np.array(observation).reshape(-1, *observation.shape))[0])

    def save_model(self, episode_reward, episode, episodes, sampling_epoc, min_reward_bar, model_name):
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
            if average_reward >= min_reward_bar or episode == episodes - 1:
                self.live_model.save(f'models/{model_name}_{episode}episode_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min_{self.discount}discount_{self.epsilon_decay}epsilondecay_{int(time.time())}.model')

# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████

agent = pong_agent(action_space = 3, observation_space = 3, replay_memory_size = replay_memory_size, epsilon = epsilon,
                   min_epsilon = min_epsilon, epsilon_decay = epsilon_decay, discount = discount, sampling_memory = sampling_memory)

environment = env()
pbar = tqdm(range(1, episodes+1), ascii = True, unit='episode')
for episode in pbar:
    episode_reward = 0

    observation = environment.set_up()
    done = False

    durata = 0

###############################################################################################
    while not done:
        action = agent.decide_the_next_action(observation)

        next_observation, reward, done = environment.step(action)
        episode_reward += reward

        agent.update_replay_memory((observation, action, reward, next_observation, done))
        observation = next_observation

        agent.train(episode = episode, how_aften_train = how_aften_train, how_aften_replace_target = how_aften_replace_target)

        if go_live and not episode % how_aften_go_live:
            environment.render()
        durata += 1
###############################################################################################

    pbar.set_description(f'epsilon {round(agent.epsilon,3)} reward {episode_reward} durata {durata}')

    agent.save_model(episode_reward, episode, episodes, sampling_epoc, min_reward_bar, model_name)

plt.plot(agent.aggr_ep_rewards['ep'], agent.aggr_ep_rewards['avg'], label = 'avg')
plt.plot(agent.aggr_ep_rewards['ep'], agent.aggr_ep_rewards['min'], label = 'min')
plt.plot(agent.aggr_ep_rewards['ep'], agent.aggr_ep_rewards['max'], label = 'max')
plt.legend(loc = 4)
plt.show()
