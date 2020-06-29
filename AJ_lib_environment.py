from PIL import Image
import cv2
import numpy as np


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
        self.x_speed = 4
        self.y_speed = 6

        self.reward_lose_game = -100
        self.reward_win_game = 100
        self.reward_hit = 5
        self.reward_still_alive = -1
        self.reward_ball_on_eye = 1

    def set_up(self):
        self.ball = element(self.size//2, self.size//2, self.x_speed, self.y_speed, self.ball_size, self.ball_size, self.size, (255, 175, 0))
        self.pad1 = element(25, self.size//2, 0, 0, 10, 50, self.size, (0, 255, 0))
        # self.pad1 = element(25, 273, 0, 0, 10, 50, self.size, (0, 255, 0))
        self.pad2 = element(self.size-25, self.size//2, 0, 0, 10, 50, self.size, (0, 0, 255))

        observation = self.ball.y, self.ball.x, self.pad1.y
        return np.array(observation)/self.size

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
        observation = self.ball.y, self.ball.x, self.pad1.y

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

        return np.array(observation)/self.size, reward, done

    def render(self):
        screen = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        self.ball.display(screen)
        self.pad1.display(screen)
        self.pad2.display(screen)

        img = Image.fromarray(screen, 'RGB')
        cv2.imshow("image", np.array(img))
        cv2.waitKey(10)
