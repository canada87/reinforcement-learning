from tqdm import tqdm
import matplotlib.pyplot as plt

from AJ_lib_battle_environment import env
from AJ_lib_agent import Agent_DQN

import streamlit as st

# ███████ ███████ ███    ██ ███████ ██ ██████  ██      ███████     ██████   █████  ██████   █████  ███    ███ ███████ ████████ ███████ ██████  ███████
# ██      ██      ████   ██ ██      ██ ██   ██ ██      ██          ██   ██ ██   ██ ██   ██ ██   ██ ████  ████ ██         ██    ██      ██   ██ ██
# ███████ █████   ██ ██  ██ ███████ ██ ██████  ██      █████       ██████  ███████ ██████  ███████ ██ ████ ██ █████      ██    █████   ██████  ███████
#      ██ ██      ██  ██ ██      ██ ██ ██   ██ ██      ██          ██      ██   ██ ██   ██ ██   ██ ██  ██  ██ ██         ██    ██      ██   ██      ██
# ███████ ███████ ██   ████ ███████ ██ ██████  ███████ ███████     ██      ██   ██ ██   ██ ██   ██ ██      ██ ███████    ██    ███████ ██   ██ ███████

stream = False

# agent
# load_pre_trained_model = '3x3_999episode__618.00max__513.10avg__105.00min_0.99discount_0.9999epsilondecay_1593110440.model'
load_pre_trained_model = None
replay_memory_size = 50_000 #dimensione massima di mosse che possono essere tenute a mente
sampling_memory = 64 #numenro di mosse usate per il training (sotto campione di replay_memory)
discount = 0.90

#game
go_live = True
save_model = False
episodes = 1 #numero massimo di partite
sampling_epoc = 100 #ogni quante epoche registrare i risultati
how_aften_go_live = 1 #ogni quante epoche fare il live di cosa succede
how_aften_replace_target = 50 #ogni quanti episodi caricare i pesi da modello live a modello target
how_aften_train = 4
min_reward_bar = 1500#valore minimo di reward per salvare i pesi del modello
model_name = 'pong_battle_8x64x4'

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
epsilon_decay = 0.999
min_epsilon = 0.05


# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████

go = st.button('go') if stream else True

if go:
    agent1 = Agent_DQN(action_space=3, observation_space=4, replay_memory_size=replay_memory_size, epsilon=epsilon,
                       min_epsilon=min_epsilon, epsilon_decay=epsilon_decay, discount=discount, sampling_memory=sampling_memory,
                       load_pre_trained_model=load_pre_trained_model)
    agent1.motion_recognition = True

    agent2 = Agent_DQN(action_space=3, observation_space=4, replay_memory_size=replay_memory_size, epsilon=epsilon,
                       min_epsilon=min_epsilon, epsilon_decay=epsilon_decay, discount=discount, sampling_memory=sampling_memory,
                       load_pre_trained_model=load_pre_trained_model)
    agent2.motion_recognition = True

    environment = env()

    image_empty = st.empty() if stream else None

    pbar = tqdm(range(1, episodes+1), ascii = True, unit='episode')
    for episode in pbar:
        episode_reward1 = 0
        episode_reward2 = 0

        observation = environment.set_up()
        done = False
        durata = 0

        ###############################################################################################
        while not done:
            action1 = agent1.decide_the_next_action(observation)
            action2 = agent2.decide_the_next_action(observation)

            next_observation, reward1, reward2, done = environment.step(action1,action2)
            episode_reward1 += reward1
            episode_reward2 += reward2

            agent1.update_replay_memory((observation, action1, reward1, next_observation, done))
            agent2.update_replay_memory((observation, action2, reward2, next_observation, done))
            observation = next_observation

            agent1.DDQN_train(episode=episode, how_aften_train=how_aften_train, how_aften_replace_target=how_aften_replace_target)
            agent2.DDQN_train(episode=episode, how_aften_train=how_aften_train, how_aften_replace_target=how_aften_replace_target)

            if go_live and not episode % how_aften_go_live:
                environment.render(image_empty)
            durata += 1
        ###############################################################################################
        agent1.update_epsilon()
        agent2.update_epsilon()

        agent1.save_model(episode_reward1, episode, episodes, sampling_epoc, min_reward_bar, model_name+'agent1', save_model)
        agent2.save_model(episode_reward2, episode, episodes, sampling_epoc, min_reward_bar, model_name+'agent2', save_model)

        pbar.set_description(f'epsilon {round(agent1.epsilon,3)} reward1 {agent1.aggr_ep_rewards["avg"][-1]} reward2 {agent2.aggr_ep_rewards["avg"][-1]} durata {durata}')


    plt.plot(agent1.aggr_ep_rewards['ep'], agent1.aggr_ep_rewards['avg'], label='avg1')
    plt.plot(agent1.aggr_ep_rewards['ep'], agent1.aggr_ep_rewards['min'], label='min1')
    plt.plot(agent1.aggr_ep_rewards['ep'], agent1.aggr_ep_rewards['max'], label='max1')
    plt.legend(loc=4)
    plt.show()

    plt.plot(agent2.aggr_ep_rewards['ep'], agent2.aggr_ep_rewards['avg'], label='avg2')
    plt.plot(agent2.aggr_ep_rewards['ep'], agent2.aggr_ep_rewards['min'], label='min2')
    plt.plot(agent2.aggr_ep_rewards['ep'], agent2.aggr_ep_rewards['max'], label='max2')
    plt.legend(loc=4)
    st.pyplot() if stream else plt.show()
