# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:19:53 2017

@author: klaudia

ADVANCED ML ICA 3
Task A: Cart-Pole problem 
Part 1:
Generate three trajectories under a uniform random policy. 
Report the episode lengths and the return from the initial state.
Part 2:
Generate 100 episodes under the above random policy, and report the 
mean and standard deviation of the episode lengths and the return 
from the starting state.

"""
# LOAD / Save: model is very simple so just run it 
# results will be the same because time seed is fixed
####################### imports ########################
import gym
import numpy as np
import matplotlib.pyplot as mtp

# load Cart pole game 
env = gym.make('CartPole-v0')
#env._max_episode_steps = max_episode_length 

######################## SETUP ##########################
# set time seeds:
np.random.seed(20) # numpy
env.seed(20)  # game environment
# frame problem:
max_len = 300 # maximum episode length
df = 0.99 # discount factor
# initialise variables 
R = [] # list for the reward per episode
C = [] # list for the number of moves per episode played
A_episodes = 3 # number of episodes set to 3 for part A and 100 for part B
B_episodes = 100
# part A ia part 1 and part Bis part 2 hence A, B suffix
########################--SOLVE--PART-1--##########################
print ("Part 1 results over",A_episodes," runs: ")    

for i in range(A_episodes):

    obs = env.reset() # initialise observation
    c = 0 # initialise counter
    temp = [] # initialise temporary reward list
  
  
    while c < (max_len)-1:
        # sample random action from uniform distrib (uniform random policy)
        act = 1*(np.random.rand()>0.5)       
        #  update 
        obs, r, done, _ = env.step(act) # obs is observation, r is reward
        c += 1 # update counter
        if done:
            r = -1 # set final reward
            temp.append(r)
            break 
        else:
            r = 0 # reward is 0 on non-terminating steps
            temp.append(r) # update temporary reward list
    
    # save number of moves per episode played
    C.append(c)
    
    # calculate reward from initial state and append to list
    r_change = sum([i*df**c for i,c in zip(temp,range(c))])
    R.append(r_change)
    
### print summary of results:
    print("Episode length was: ", c)
    print('The corresponding return from initial state (i.e.reward) was:', round(r_change,5))

########################--SOLVE--PART-2--##########################
# repeat as per part 2 but this time repeat over 100 runs and incl. plots 


for i in range(B_episodes):

    observation = env.reset() # initialise observation
    c = 0 # initialise counter
    temp = [] # initialise temporary reward list
  
  
    while c < (max_len)-1:
        # sample random action from uniform distrib (uniform random policy)
        act = 1*(np.random.rand()>0.5)       
        #  update 
        obs, r, done, info = env.step(act) # obs is observation, r is reward
        c += 1 # update counter
        if done:
            r = -1 # set final reward
            temp.append(r)
            break 
        else:
            r = 0 # reward is 0 on non-terminating steps
            temp.append(r) # update temporary reward list
    
    # save number of moves per episode played
    C.append(c)
    
    # calculate reward from initial state and append to list
    r_change = sum([i*df**c for i,c in zip(temp,range(c))])
    R.append(r_change)
    
### print summary of results:
print ("-----------------------------------------------------------------")   
print ("Part 2 results over",B_episodes," runs: ")    
print("Mean episode length was: ", np.mean(C), "and the corresponding standard deviation was: ", np.std(C) )
print('Mean return from initial state was:', np.mean(R), "and the corresponding standard deviation was: ", np.std(R) )



# MAKE PLOTS of results
print ("\nPart 2 Supporting plots: ")    
mtp.figure(1)
mtp.plot(C)
mtp.title('Mean episode length vs Index', fontsize = 12)
mtp.ylabel('Mean episode length')
mtp.xlabel('Episode number')

mtp.figure(2)
mtp.plot(R)
mtp.title('Mean return from initial state vs Index', fontsize = 12)
mtp.ylabel('Mean return from initial state')
mtp.xlabel('Episode number')