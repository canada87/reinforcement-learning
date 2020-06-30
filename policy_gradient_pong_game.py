from tqdm import tqdm
import matplotlib.pyplot as plt

from AJ_lib_environment import env
from AJ_lib_agent import Agent_policy_gradient


# ███████ ███████ ███    ██ ███████ ██ ██████  ██      ███████     ██████   █████  ██████   █████  ███    ███ ███████ ████████ ███████ ██████  ███████
# ██      ██      ████   ██ ██      ██ ██   ██ ██      ██          ██   ██ ██   ██ ██   ██ ██   ██ ████  ████ ██         ██    ██      ██   ██ ██
# ███████ █████   ██ ██  ██ ███████ ██ ██████  ██      █████       ██████  ███████ ██████  ███████ ██ ████ ██ █████      ██    █████   ██████  ███████
#      ██ ██      ██  ██ ██      ██ ██ ██   ██ ██      ██          ██      ██   ██ ██   ██ ██   ██ ██  ██  ██ ██         ██    ██      ██   ██      ██
# ███████ ███████ ██   ████ ███████ ██ ██████  ███████ ███████     ██      ██   ██ ██   ██ ██   ██ ██      ██ ███████    ██    ███████ ██   ██ ███████

#game
load_pre_trained_policy = None
load_pre_trained_predict = None
go_live = True
save_model = True
episodes = 10000 #numero massimo di partite
how_aften_go_live = 2000 #ogni quante epoche fare il live di cosa succede
sampling_epoc = 100 #ogni quante epoche registrare i risultati
min_reward_bar = 1500#valore minimo di reward per salvare i pesi del modello
model_name = 'pong_policy_8x64x4'

# Exploration settings
epsilon_decay = 0.99


# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████

agent = Agent_policy_gradient(action_space=3, observation_space=3, epsilon_decay=epsilon_decay,
                              load_pre_trained_policy=load_pre_trained_policy, load_pre_trained_predict = load_pre_trained_predict)

environment = env()

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
        episode_reward += reward

        agent.update_memory(observation, action, reward)
        observation = next_observation

        agent.policy_gradient_learning()

        if go_live and not episode % how_aften_go_live:
            environment.render()
        durata += 1
    ###############################################################################################

    agent.save_model(episode_reward, episode, episodes, sampling_epoc, min_reward_bar, model_name, save_model)
    pbar.set_description(f'epsilon {round(agent.epsilon_decay,3)} reward {agent.aggr_ep_rewards["avg"][-1]} durata {durata}')


plt.plot(agent.aggr_ep_rewards['ep'], agent.aggr_ep_rewards['avg'], label='avg')
plt.plot(agent.aggr_ep_rewards['ep'], agent.aggr_ep_rewards['min'], label='min')
plt.plot(agent.aggr_ep_rewards['ep'], agent.aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()
