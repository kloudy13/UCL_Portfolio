# -*- coding: utf-8 -*-
"""

@author: klaudia
Problem B atari games 
Part 1:
Report the score and frame counts from the three games under a random policy, 
evaluated on 100 episodes. Report both the average and the standard deviation.
"""

# ============================================================================
# ========================== IMPORTANT !!!! ==================================
# SETUP to save/load
Train = False # use this setting to load the model
# if Train = True model will train and save results 
# if Train = False summary results will be displayed

# CHOSE GAME 
# only workis for Boxing and MsPacman, to load relevant model chose between
# environment name 'Box' or 'Pacman' or'Pong'
env_name = 'Pacman'#'Box'#'Pong' '

####################### imports ########################
import gym
import numpy as np
import tensorflow as tf

######################## SETUP ##########################

# init. vars
start_ep = 100  # number of start episodes to be used
df = 0.99 # discount factor

# Select number of possible actions based on game 
# and load desired game environ.:
box_poss_act = 18
pac_poss_act = 9
pong_poss_act = 6

if env_name == 'Box':
    poss_act = box_poss_act
    env = gym.make('Boxing-v3')
elif env_name == 'Pacman':
    poss_act = pac_poss_act
    env = gym.make('MsPacman-v3')
else:
    poss_act = pong_poss_act
    env = gym.make('Pong-v3')
np.set_printoptions(threshold=np.nan)

# for load and save:
folder = "models/"
file = "PartB1_game:"+env_name
file_save = folder + file
file_N = file + '.npz' # for numpy save/load variables

# set time seeds:
np.random.seed(20)
tf.set_random_seed(20)
env.seed(20) 

##################################--MAIN--###################################
if Train == True:
    print("start training game: ",env_name)
    # init. more vars.
    rewOut =[] # to store final reward
    ep_len =[] # to store episode length
    
    for i in range(start_ep):
        done = False
        # init. counters:
        cc = 0 # episode length counter
        sum_r = 0.0 # sum of episode rewards
        env.reset() # reset environ
        
        while not done:
            cc += 1
            action = env.action_space.sample() # random policy
            _, rew, done, _ = env.step(action)
            rew = np.clip(rew, -1, 1) # clip reward
            sum_r += (df ** (cc))*rew       
            if done:
                break
            
        rewOut.append(sum_r)
        ep_len.append(cc)
        if i%10==0:
            print(' episode:',i, "corresonding return is ",rewOut[i]," and length is ",cc )
    
    # Save results    
    np.savez(file_N, rewOut = rewOut, ep_len = ep_len)

###############################----RESULTS---##################################
#  if not training load saved results:
else: 
        loaded = np.load(file_N)
        print("Model Loaded\n") 
        print("Paty B 1, game: ",env_name)
        rewOut = loaded['rewOut']
        ep_len = loaded['ep_len']
        
# calc mean and dtandard deviation of score and frames:
ave_score = np.mean(rewOut)
sd_score = np.std(rewOut)
ave_frames = np.mean(ep_len)
sd_frames = np.std(ep_len)
# display results:
print("The Mean Score (i.e discounted reward) is:",ave_score," and the standard deviation is:",sd_score)
print("The Mean Frames (i.e episode length) are:",ave_frames," and their standard deviation is:",sd_frames)

