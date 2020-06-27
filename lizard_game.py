from tqdm import tqdm
import matplotlib.pyplot as plt

from lizard_env import env
from AJ_lib_agent import Agent


# ███████ ███████ ███    ██ ███████ ██ ██████  ██      ███████     ██████   █████  ██████   █████  ███    ███ ███████ ████████ ███████ ██████  ███████
# ██      ██      ████   ██ ██      ██ ██   ██ ██      ██          ██   ██ ██   ██ ██   ██ ██   ██ ████  ████ ██         ██    ██      ██   ██ ██
# ███████ █████   ██ ██  ██ ███████ ██ ██████  ██      █████       ██████  ███████ ██████  ███████ ██ ████ ██ █████      ██    █████   ██████  ███████
#      ██ ██      ██  ██ ██      ██ ██ ██   ██ ██      ██          ██      ██   ██ ██   ██ ██   ██ ██  ██  ██ ██         ██    ██      ██   ██      ██
# ███████ ███████ ██   ████ ███████ ██ ██████  ███████ ███████     ██      ██   ██ ██   ██ ██   ██ ██      ██ ███████    ██    ███████ ██   ██ ███████

# agent
load_pre_trained_model = 'lizard_2x4_9999episode_11max_-41.54avg_-304min_0.99discount_0.99999epsilondecay_1593186815.model'
# load_pre_trained_model = None
replay_memory_size = 50_000 #dimensione massima di mosse che possono essere tenute a mente
sampling_memory = 64 #numenro di mosse usate per il training (sotto campione di replay_memory)
discount = 0.99

#game
go_live = False
save_model = True
episodes = 5_000 #numero massimo di partite
sampling_epoc = 100 #ogni quante epoche registrare i risultati
how_aften_go_live = 1000 #ogni quante epoche fare il live di cosa succede
how_aften_replace_target = 50 #ogni quanto caricare i pesi da modello live a modello target
how_aften_train = 4
min_reward_bar = 5#valore minimo di reward per salvare i pesi del modello
model_name = 'lizard_fixed_2x4'

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
epsilon_decay = 0.99999
min_epsilon = 0.05


# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████

agent = Agent(action_space=4, observation_space=8, replay_memory_size=replay_memory_size, epsilon=epsilon,
                   min_epsilon=min_epsilon, epsilon_decay=epsilon_decay, discount=discount, sampling_memory=sampling_memory,
                   load_pre_trained_model=load_pre_trained_model)

environment = env()

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

        if durata >= 100:
            done = True
            reward = -10

        episode_reward += reward

        agent.update_replay_memory((observation, action, reward, next_observation, done))
        observation = next_observation

        # agent.DDQN_train(episode=episode, how_aften_train=how_aften_train, how_aften_replace_target=how_aften_replace_target)
        agent.DQN_train(episode=episode, how_aften_train=how_aften_train)

        if go_live and not episode % how_aften_go_live:
            print('action', action, 'reward', reward)
            environment.render()
        durata += 1
    ##############################################################################################

    agent.save_model(episode_reward, episode, episodes, sampling_epoc, min_reward_bar, model_name, save_model)
    pbar.set_description(f'epsilon {round(agent.epsilon,3)} reward {agent.aggr_ep_rewards["avg"][-1]} durata {durata}')

plt.plot(agent.aggr_ep_rewards['ep'], agent.aggr_ep_rewards['avg'], label='avg')
plt.plot(agent.aggr_ep_rewards['ep'], agent.aggr_ep_rewards['min'], label='min')
plt.plot(agent.aggr_ep_rewards['ep'], agent.aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()
