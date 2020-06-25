import numpy as np

vettore = [i for i in range(40,60)]

indices = np.random.choice(range(20), size=5)

state_sample = np.array([vettore[i] for i in indices])
print(state_sample)
