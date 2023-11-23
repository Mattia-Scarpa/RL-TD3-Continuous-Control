# importing required libraries

# main packages
import stat
from tkinter import E
import numpy as np
import torch

from collections import deque
import matplotlib.pyplot as plt


# import environment
from unityagents import UnityEnvironment

# import agent
from agent import TD3Agent


# parameters
PRINT_EVERY = 10
MAX_T = 1000
N_EPISODES = 1000

EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.999


# TD3 function

def td3(env, brain_name, agent, n_episodes=N_EPISODES, max_t=None, print_every=PRINT_EVERY, num_agents=1, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY):

    scores = []
    scores_window = deque(maxlen=print_every)
    eps = eps_start
    best_score = 30.

    for i_episode in range(1, n_episodes+1):

        state = env.reset(train_mode=True)[brain_name].vector_observations[0]
        score = 0
        thresh_count = 0

        if max_t is None:
            max_t = 1e7
        
        for t in range(int(max_t)):
            action = agent.act(state, eps)

            env_info = env.step(action)[brain_name]

            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward

            if done:
                break   

        if i_episode > 100 and score > 30.0:
            thresh_count += 1 
        else:
            thresh_count = 0
        if thresh_count > 100:      # stop if agent is solved stabilly for 100 episodes
            break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= best_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            best_score = np.mean(scores_window)
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

    return scores


# testing function

def test(env, brain_name, agent, n_episodes=100, max_t=1000, num_agents=1):
    
        scores = []
    
        for i_episode in range(1, n_episodes+1):
    
            state = env.reset(train_mode=True)[brain_name].vector_observations[0]
            score = 0
    
            for t in range(max_t):
                action = agent.act(state)
    
                env_info = env.step(action)[brain_name]
    
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]
    
                state = next_state
                score += reward
    
                if done:
                    break
    
            scores.append(score)
    
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores)))
    
        return scores

# main body

def main():
    
    # initializing environment
    env = UnityEnvironment(file_name='Reacher_Linux_single/Reacher.x86_64')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    action_size = brain.vector_action_space_size
    print('Action max: ', brain.vector_action_descriptions)
    state_size = brain.vector_observation_space_size

    print('Observation space: ', state_size)
    print('Action space: ', action_size)

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents: ', num_agents)

    # initialize agent
    agent = TD3Agent(state_size=state_size, action_size=action_size, a_max=1, seed=0)

    scores = td3(env, brain_name, agent)

    print('Training finished.')

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores, color='blue')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('Training average scores over 100 episodes')
    plt.savefig('training_scores.png')
    

    # load the weights from file
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

    # test the agent
    scores = test(env, brain_name, agent)

    print('Total test score (averaged over agents) for 100 episodes: {}'.format(np.mean(scores)))

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores, color='blue')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('Test scores per episode')
    plt.savefig('test_scores.png')
    
    # close the environment
    env.close()

# executing main function
if __name__ == '__main__':
    main()