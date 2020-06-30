import numpy as np
from sklearn.preprocessing import OneHotEncoder

action = [0,1,4,3,1,0,1,0]

action_space = 5

actions = np.zeros([len(action), action_space])
actions[np.arange(len(action)), action] = 1

actions

enc = OneHotEncoder()
enc.fit_transform(action).toarray()
