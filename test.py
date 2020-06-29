from collections import deque
import numpy as np
import random
import itertools

replay_memory = deque(maxlen = 50)
replay_memory_sample = deque(maxlen = 15)

sampling_memory = 15

stack_len = 3
batch_len = (sampling_memory//stack_len) # 5


for i in range(50):
    replay_memory.append(i)
# print(replay_memory)
#
# print(list(itertools.islice(replay_memory,20,25)))

# print(replay_memory.shape())



    if len(replay_memory)>15:
        rand_index = random.sample([i for i in range(len(replay_memory) - stack_len)], batch_len)

        minibatch3 = np.zeros((len(rand_index)*stack_len, ))
        count = 0

        for i in rand_index:
            # print(i)
            # minibatch3 = list(itertools.islice(replay_memory,i,i+stack_len))

            print( list(itertools.islice(replay_memory,i,i+stack_len))[0])

            minibatch3[count:count+stack_len] = list(itertools.islice(replay_memory,i,i+stack_len))

            replay_memory_sample.append(list(itertools.islice(replay_memory,i,i+stack_len)))
            count += stack_len

            # for j in range(stack_len):
            #     replay_memory_sample.append([i+j])

        # print(replay_memory_sample)
        # print(minibatch3)
        # print()
