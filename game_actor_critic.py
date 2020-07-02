from tqdm import tqdm
import matplotlib.pyplot as plt

from AJ_lib_environment_pong import env_pong
from AJ_lib_environment_lizard import env_lizard
from AJ_lib_agent import Actor_critic

import streamlit as st

# ███████ ███████ ███    ██ ███████ ██ ██████  ██      ███████     ██████   █████  ██████   █████  ███    ███ ███████ ████████ ███████ ██████  ███████
# ██      ██      ████   ██ ██      ██ ██   ██ ██      ██          ██   ██ ██   ██ ██   ██ ██   ██ ████  ████ ██         ██    ██      ██   ██ ██
# ███████ █████   ██ ██  ██ ███████ ██ ██████  ██      █████       ██████  ███████ ██████  ███████ ██ ████ ██ █████      ██    █████   ██████  ███████
#      ██ ██      ██  ██ ██      ██ ██ ██   ██ ██      ██          ██      ██   ██ ██   ██ ██   ██ ██  ██  ██ ██         ██    ██      ██   ██      ██
# ███████ ███████ ██   ████ ███████ ██ ██████  ███████ ███████     ██      ██   ██ ██   ██ ██   ██ ██      ██ ███████    ██    ███████ ██   ██ ███████

game = 2 #1 lizard 2 pong
stream = True

#game
load_pre_trained_critic = None
load_pre_trained_policy = None
load_pre_trained_actor = None

go_live = True
save_model = False
episodes = 1 #numero massimo di partite
how_aften_go_live = 1 #ogni quante epoche fare il live di cosa succede
sampling_epoc = 100 #ogni quante epoche registrare i risultati
min_reward_bar = 1500#valore minimo di reward per salvare i pesi del modello
model_name = 'lizard_actor_8x64x4'

# Exploration settings
epsilon_decay = 0.99


# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████

go = st.button('go') if stream else True

if go:
    num_action = 3 if game == 2 else 4
    num_observation = 3 if game == 2 else 8

    agent = Actor_critic(action_space=num_action, observation_space=num_observation, epsilon_decay=epsilon_decay,
                                  load_pre_trained_critic=load_pre_trained_critic, load_pre_trained_policy=load_pre_trained_policy,
                                  load_pre_trained_actor=load_pre_trained_actor)
    agent.learning_rate_actor = 0.00001
    agent.learning_rate_critic= 0.00005

    if game == 1:
        environment = env_lizard()
        environment.fixed_initial_pos = True
        environment.active_adversary_movents = True
    else:
        environment = env_pong()

    image_empty = st.empty() if stream else None

    pbar = tqdm(range(1, episodes+1), ascii = True, unit='episode')
    for episode in pbar:
        episode_reward = 0

        observation = environment.set_up()
        done = False
        durata = 0
        ##############################################################################################
        while not done:
            action = agent.decide_the_next_action(observation)

            next_observation, reward, done = environment.step(action)
            if durata >= 100 and game == 1:
                done = True
                reward = -10

            agent.training(observation, action, reward, next_observation, done)

            episode_reward += reward
            observation = next_observation

            if go_live and not episode % how_aften_go_live:
                environment.render(image_empty)
            durata += 1
        ##############################################################################################
        agent.save_model(episode_reward, episode, episodes, sampling_epoc, min_reward_bar, model_name, save_model)
        pbar.set_description(f'epsilon {round(agent.epsilon_decay,3)} reward {agent.aggr_ep_rewards["avg"][-1]} durata {durata}')

    plt.plot(agent.aggr_ep_rewards['ep'], agent.aggr_ep_rewards['avg'], label='avg')
    plt.plot(agent.aggr_ep_rewards['ep'], agent.aggr_ep_rewards['min'], label='min')
    plt.plot(agent.aggr_ep_rewards['ep'], agent.aggr_ep_rewards['max'], label='max')
    plt.legend(loc=4)
    st.pyplot() if stream else plt.show()
