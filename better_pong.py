import numpy as np
# import keras.backend.tensorflow_backend as backend
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
# from keras.optimizers import Adam
# from keras.callbacks import TensorBoard
# import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2

class element:
    def __init__(self, x, y, size):
        self.size = size
        self.x = x
        self.y = y

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def action(self, choice):
        if choice == 0:
            self.move(x=0, y=20)
        elif choice == 1:
            return self.move(x=0, y=-20)
        elif choice == 2:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):
        done = False
        # If no value for x, move randomly
        if not x:
            self.x = self.x
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y = self.y
        else:
            self.y += y

        # boundaries
        if self.x < 0:
            self.x = 0
            done = True
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
            done = True
        elif self.y > self.size-1:
            self.y = self.size-1

        return done
class Env:
    SIZE = 100
    PLAYER_N = 1  # player key in dict
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def reset(self):
        self.player_a = element(self.SIZE//2, self.SIZE//2, self.SIZE)
        self.player_b = element(self.SIZE//2, self.SIZE//3, self.SIZE)

    def step(self, action):
        return self.player_a.action(action)

    def render(self):
        img = self.get_image()
        img = img.resize((500, 500))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1000)

    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.player_a.x][self.player_a.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')
        return img

env = Env()
env.reset()

done = False

while not done:
    env.render()
    done = env.step(1)


# env.step(1)
# env.render()
# env.step(1)
# env.render()
# env.step(1)
# env.render()


# EPISODES = 3
#
# for episode in tqdm(range(1, EPISODES+1), ascii = True, unit='episode'):
#     done = False
#     while not done:
#         new_state, reward, done = env.step(1)
#         new_state, reward, done = env.step(1)
#         new_state, reward, done = env.step(1)
#         env.render()
#
