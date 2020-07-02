from tqdm import tqdm
import matplotlib.pyplot as plt

from AJ_lib_environment_lizard import env_lizard
from AJ_lib_environment_pong import env_pong
from AJ_lib_agent import Agent_DQN


# ███████ ███████ ███    ██ ███████ ██ ██████  ██      ███████     ██████   █████  ██████   █████  ███    ███ ███████ ████████ ███████ ██████  ███████
# ██      ██      ████   ██ ██      ██ ██   ██ ██      ██          ██   ██ ██   ██ ██   ██ ██   ██ ████  ████ ██         ██    ██      ██   ██ ██
# ███████ █████   ██ ██  ██ ███████ ██ ██████  ██      █████       ██████  ███████ ██████  ███████ ██ ████ ██ █████      ██    █████   ██████  ███████
#      ██ ██      ██  ██ ██      ██ ██ ██   ██ ██      ██          ██      ██   ██ ██   ██ ██   ██ ██  ██  ██ ██         ██    ██      ██   ██      ██
# ███████ ███████ ██   ████ ███████ ██ ██████  ███████ ███████     ██      ██   ██ ██   ██ ██   ██ ██      ██ ███████    ██    ███████ ██   ██ ███████

game = 1 #1 lizard 2 pong

# agent
# load_pre_trained_model = '3x3_999episode__618.00max__513.10avg__105.00min_0.99discount_0.9999epsilondecay_1593110440.model'
load_pre_trained_model = None
replay_memory_size = 50_000 #dimensione massima di mosse che possono essere tenute a mente
sampling_memory = 64 #numenro di mosse usate per il training (sotto campione di replay_memory)
discount = 0.99

#game
go_live = True
save_model = False
episodes = 10000 #numero massimo di partite
sampling_epoc = 100 #ogni quante epoche registrare i risultati
how_aften_go_live = 5000 #ogni quante epoche fare il live di cosa succede
how_aften_replace_target = 50 #ogni quanti episodi caricare i pesi da modello live a modello target
how_aften_train = 4
min_reward_bar = 1500#valore minimo di reward per salvare i pesi del modello
model_name = 'pong_motion_8x64x4'

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
epsilon_decay = 0.999
min_epsilon = 0.05


# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████

num_action = 3 if game == 2 else 4
num_observation = 3 if game == 2 else 8

agent = Agent_DQN(action_space=num_action, observation_space=num_observation, replay_memory_size=replay_memory_size, epsilon=epsilon,
                   min_epsilon=min_epsilon, epsilon_decay=epsilon_decay, discount=discount, sampling_memory=sampling_memory,
                   load_pre_trained_model=load_pre_trained_model)
agent.motion_recognition = True

if game == 1:
    environment = env_lizard()
    environment.fixed_initial_pos = True
    environment.active_adversary_movents = True
else:
    environment = env_pong()

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
        if durata >= 100 and game == 1:
            done = True
            reward = -10

        episode_reward += reward

        agent.update_replay_memory((observation, action, reward, next_observation, done))
        observation = next_observation

        agent.DDQN_train(episode=episode, how_aften_train=how_aften_train, how_aften_replace_target=how_aften_replace_target)
        # agent.DQN_train(episode=episode, how_aften_train=how_aften_train)


        if go_live and not episode % how_aften_go_live:
            environment.render()
        durata += 1
    ###############################################################################################
    agent.update_epsilon()

    agent.save_model(episode_reward, episode, episodes, sampling_epoc, min_reward_bar, model_name, save_model)
    pbar.set_description(f'epsilon {round(agent.epsilon,3)} reward {agent.aggr_ep_rewards["avg"][-1]} durata {durata}')


plt.plot(agent.aggr_ep_rewards['ep'], agent.aggr_ep_rewards['avg'], label='avg')
plt.plot(agent.aggr_ep_rewards['ep'], agent.aggr_ep_rewards['min'], label='min')
plt.plot(agent.aggr_ep_rewards['ep'], agent.aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()
