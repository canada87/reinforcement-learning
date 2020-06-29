from PIL import Image
import cv2
import numpy as np
import random


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
        ball_pixels[self.x - self.x_size: self.x + self.x_size, self.y - self.y_size: self.y + self.y_size] = self.color
        background += ball_pixels

    def action(self, choice):
        reward = 0
        if choice == 0:
            reward = self.move(y=self.y_size*2, x = 0)
        elif choice == 1:
            reward = self.move(y=-self.y_size*2, x = 0)
        elif choice == 2:
            reward = self.move(x=self.x_size*2, y = 0)
        elif choice == 3:
            reward = self.move(x=self.x_size*2, y = 0)
        return reward

    def move(self, x, y):
        reward = 0
        self.y += y
        if self.y > self.w_size - self.y_size:
            self.y = self.w_size - self.y_size
            reward = -2
        if self.y < self.y_size:
            self.y = self.y_size
            reward = -2

        self.x += x
        if self.x > self.w_size - self.x_size:
            self.x = self.w_size - self.x_size
            reward = -2
        if self.x < self.x_size:
            self.x = self.x_size
            reward = -2
        return reward

    def not_overimpose(self, other):
        if self.x == other.x and self.y == other.y:
            self.action(np.random.randint(0, 4))

    def is_overimpose(self, other):
        if self.x == other.x and self.y == other.y:
            return 1
        else:
            return 0

    def hit(self, branco, cavalletta, falco):
        reward = -1
        done = False
        if self.x == branco.x and self.y == branco.y:
            reward = 10
            done = True
        if self.x == cavalletta.x and self.y == cavalletta.y:
            reward = 1
        if self.x == falco.x and self.y == falco.y:
            reward = -10
            done = True
        return reward, done


# ███████ ███    ██ ██    ██ ██ ██████   ██████  ███    ██ ███    ███ ███████ ███    ██ ████████
# ██      ████   ██ ██    ██ ██ ██   ██ ██    ██ ████   ██ ████  ████ ██      ████   ██    ██
# █████   ██ ██  ██ ██    ██ ██ ██████  ██    ██ ██ ██  ██ ██ ████ ██ █████   ██ ██  ██    ██
# ██      ██  ██ ██  ██  ██  ██ ██   ██ ██    ██ ██  ██ ██ ██  ██  ██ ██      ██  ██ ██    ██
# ███████ ██   ████   ████   ██ ██   ██  ██████  ██   ████ ██      ██ ███████ ██   ████    ██


class env:

    def __init__(self):
        self.size = 500
        self.num_square = 5
        resto = self.size%self.num_square
        self.size = self.size - resto
        self.fixed_initial_pos = True
        self.active_adversary_movents = False

    def set_up(self):
        def random_start(x_poses, y_poses):
            x_pos = random.choice(x_poses)
            y_pos = random.choice(y_poses)
            ix = x_poses.index(x_pos)
            iy = y_poses.index(y_pos)
            x_poses.pop(ix)
            y_poses.pop(iy)
            return x_pos, y_pos

        x_poses =  [i for i in range(self.num_square)]
        y_poses =  [i for i in range(self.num_square)]

        if self.fixed_initial_pos:
            x_pos, y_pos = self.size//(self.num_square * 2), self.size//(self.num_square * 2)
        else:
            # x_pos, y_pos = 0,0
            # ix = x_poses.index(x_pos)
            # iy = y_poses.index(y_pos)
            # x_poses.pop(ix)
            # y_poses.pop(iy)
            x_pos, y_pos = random_start(x_poses, y_poses)
            x_pos = self.size//(self.num_square * 2)+ self.size//(self.num_square)*x_pos
            y_pos = self.size//(self.num_square * 2)+ self.size//(self.num_square)*y_pos
        self.lizard = element(x_pos, y_pos, 0, 0, self.size//(self.num_square * 2), self.size//(self.num_square * 2), self.size, (255, 175, 0))#magenta

        if self.fixed_initial_pos:
            x_pos, y_pos = self.size//(self.num_square * 2)+ self.size//(self.num_square)*4, self.size//(self.num_square * 2)+ self.size//(self.num_square)*4
        else:
            x_pos, y_pos = random_start(x_poses, y_poses)
            x_pos = self.size//(self.num_square * 2)+ self.size//(self.num_square)*x_pos
            y_pos = self.size//(self.num_square * 2)+ self.size//(self.num_square)*y_pos
        self.branco = element(x_pos, y_pos, 0, 0, self.size//(self.num_square * 2), self.size//(self.num_square * 2), self.size, (255, 0, 0))#blu

        if self.fixed_initial_pos:
            x_pos, y_pos = self.size//(self.num_square * 2)+ self.size//(self.num_square)*0, self.size//(self.num_square * 2)+ self.size//(self.num_square)*2
        else:
            x_pos, y_pos = random_start(x_poses, y_poses)
            x_pos = self.size//(self.num_square * 2)+ self.size//(self.num_square)*x_pos
            y_pos = self.size//(self.num_square * 2)+ self.size//(self.num_square)*y_pos
        self.cavalletta = element(x_pos, y_pos, 0, 0, self.size//(self.num_square * 2), self.size//(self.num_square * 2), self.size, (0, 255, 0))#verde

        if self.fixed_initial_pos:
            x_pos, y_pos = self.size//(self.num_square * 2)+ self.size//(self.num_square)*2, self.size//(self.num_square * 2)+ self.size//(self.num_square)*2
        else:
            x_pos, y_pos = random_start(x_poses, y_poses)
            x_pos = self.size//(self.num_square * 2)+ self.size//(self.num_square)*x_pos
            y_pos = self.size//(self.num_square * 2)+ self.size//(self.num_square)*y_pos
        self.falco = element(x_pos, y_pos, 0, 0, self.size//(self.num_square * 2), self.size//(self.num_square * 2), self.size, (0, 0, 255))#rosso

        observation = self.lizard.x, self.lizard.y, self.branco.x, self.branco.y, self.cavalletta.x, self.cavalletta.y, self.falco.x, self.falco.y
        return np.array(observation)/self.size

    def step(self, action):
        reward_wall = self.lizard.action(action)

        # #adversary movement
        if self.active_adversary_movents:
            self.cavalletta.action(np.random.randint(0, 4))
            self.branco.action(np.random.randint(0, 4))
            self.falco.action(np.random.randint(0, 4))

            done_over = False
            while not done_over:
                count = 0
                count += self.cavalletta.is_overimpose(self.branco)
                count += self.cavalletta.is_overimpose(self.falco)
                count += self.branco.is_overimpose(self.cavalletta)
                count += self.branco.is_overimpose(self.falco)
                count += self.falco.is_overimpose(self.cavalletta)
                count += self.falco.is_overimpose(self.branco)
                self.cavalletta.not_overimpose(self.branco)
                self.cavalletta.not_overimpose(self.falco)
                self.branco.not_overimpose(self.cavalletta)
                self.branco.not_overimpose(self.falco)
                self.falco.not_overimpose(self.cavalletta)
                self.falco.not_overimpose(self.branco)
                if count == 0:
                    done_over = True
                else:
                    done_over = False

        observation = self.lizard.x, self.lizard.y, self.branco.x, self.branco.y, self.cavalletta.x, self.cavalletta.y, self.falco.x, self.falco.y
        reward, done = self.lizard.hit(self.branco, self.cavalletta, self.falco)
        reward += reward_wall
        return np.array(observation)/self.size, reward, done

    def render(self):
        screen = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        self.lizard.display(screen)
        self.falco.display(screen)
        self.branco.display(screen)
        self.cavalletta.display(screen)

        img = Image.fromarray(screen, 'RGB')
        cv2.imshow("image", np.array(img))
        cv2.waitKey(500)
