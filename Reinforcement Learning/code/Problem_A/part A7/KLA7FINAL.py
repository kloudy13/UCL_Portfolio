# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:39:08 2017

@author: klaudia
Task A: Cart-Pole problem  
Part 7 task:
Start with the network from the previous part and add a target network, 
copying the current network parameters to the target network every 5 episodes. 
Train and compare. (Plot the test performance and bellman loss every 20 steps).
"""
# ============================================================================
# ========================== IMPORTANT !!!! ==================================
# SETUP to save/load
Train =  True#False # use this setting to load the model
# if Train = True model will train and save results 
# if Train = False model will LOAD and summary results will be displayed
# ============================================================================
####################### imports ########################
import gym
import numpy as np
import matplotlib.pyplot as mtp
import tensorflow as tf
import os.path
import random

######################## SETUP ##########################

# frame problem:
max_len = 300 # maximum episode length
df = 0.99 # discount factor
# initialise variables 
episodes = 2000 # number of episodes 
eval_ep_N = 20 # no. of  episodes for evaluation 
batch_size = 1000 # chosen after trial and error 
learn_rate = 0.1  # learning rate
state_size = 4   # size of state vector
poss_action = 2  # number of possible actions
hidden_size = 100  # size of hidden layer
epsilon = 0.05 # set constant for e_greedy policy
buff_size = 50000   # size of the replay buffer 
start_ep = 1200      # number of start episodes to be used
# for load and save:
folder = "models/"
file = "PartA7_ "+"LR:" + str(learn_rate)
file_save = folder + file


# load Cart pole game 
env = gym.make('CartPole-v0') # switch back to v0 as have found a way to increase max length (look below)
np.set_printoptions(threshold=np.nan)
# set max length to 300 as per queston requirement:
env._max_episode_steps = max_len

# set time seeds:
np.random.seed(20)
tf.set_random_seed(20)
env.seed(20) 

############################### FUNCTIONS #################################

def MYgreed(epsilon, Q):
    test = np.random.rand()
    if (test <= epsilon):
        return 1*(test >0.5)
    else:
        return Q

def MakeBatch2(data):
#make batches of data coming in the form : [old_observation, observation, action, reward, done]
    
    # create arrays with first data instance written to them 
    batch_state_now = data[0][0] # current batch state
    batch_state_next = data[0][1] # next batch state
    batch_act = data[0][2] # batch action
    batch_r = data[0][3] # batch reward
    batch_done = data[0][4]
    # add consecutive data instances to create batch
    for i in range(1,len(data)):
        batch_state_now = np.vstack((batch_state_now ,data[i][0]))
        batch_state_next = np.vstack((batch_state_next ,data[i][1]))
        batch_act = np.vstack((batch_act ,data[i][2]))
        batch_r = np.vstack((batch_r ,data[i][3]))
        batch_done = np.vstack((batch_done, data[i][4]))
    return batch_state_now, batch_state_next, batch_act, batch_r, batch_done
    
def MakeBuffer(episodes):
# function to make random uniform data for the episodes   
    data = RunBuffer()
    for i_episode in range(episodes):
        done = False 
        obs = env.reset() # initialise observation
        c = 0 # initialise counter
        while c < (max_len) -1:
            obs1 = obs
            # random action from uniform distrib (uniform random policy)
            action = (np.random.rand()>0.5)*1
           #  update 
            obs, rew, done, _ = env.step(action)
            c += 1 # update counter
            if done:
                rew = -1 # set final reward
                data.Plus([obs1.reshape([1,4]), obs.reshape([1,4]), action, rew, (not done)*1])
                break 
            else:
                rew = 0 # reward is 0 on non-terminating steps
            data.Plus([obs1.reshape([1,4]), obs.reshape([1,4]), action, rew, (not done)*1])
    return data

class RunBuffer(object):
# class will hold the experineces (to make the experinec replay buffer)
    def __init__(self, buff_size = buff_size):
        self.sth = []      # array to store    
        self.buff_size = buff_size # buffer size
    
    def RandBuff(self, batch_size):
        # sample random batch from the replay buffer (size = batch_size)
        return np.reshape(np.array(random.sample(self.sth, batch_size)), [batch_size, 5])
       
    def Plus(self,state): # pass in the state vector
        # add experinece to buffer
        # of buffer is full remove fisrt data instance and stack new instance:
        if len(self.sth)+len(state) > self.buff_size:
        # take out first data instance from buffer stack
            self.sth[0:(len(state)+len(self.sth))-self.buff_size] = []
        # append new data instance
        self.sth.append(state)
        
        
##################################--MAIN--###################################

tf.reset_default_graph() # reset graph 

# initialise placeholders for state vectors:
state_now = tf.placeholder(tf.float32, shape = [None,state_size], name ='state_now')
state_next = tf.placeholder(tf.float32, shape = [None,state_size], name ='state_next')
# initialise placeholder for actions:                
act = tf.placeholder(tf.int32, shape = [None,poss_action], name ='action')
# initialise placeholder for reward:                      
r = tf.placeholder(tf.float32, shape = [None,1], name ='reward')
# need to make the 'done' variable into tensor so as to use the expeinece buffer
isdone = tf.placeholder(tf.float32, shape = [None,1], name = 'isdone')
# target weights 1st layer
w_tgt_1 = tf.placeholder(tf.float32, shape=[state_size, hidden_size], name ='weight1')
# target weights hidden layer                            
w_tgt_h = tf.placeholder(tf.float32, shape=[hidden_size, poss_action], name ='weight_h')

 # Define weights for NN (without bais term)
# weights initialiser:
init = tf.contrib.layers.xavier_initializer(uniform=True, seed = 20, dtype=tf.float32)

w = {
    'Layer1': tf.get_variable('w1', shape=[state_size, hidden_size], initializer = init), 
    'Layer2': tf.get_variable('w2', shape=[hidden_size, poss_action], initializer = init)}

# define network with hidden ReLU units 
# obtain present Q estimate:
hidden1 = tf.matmul(state_now, w['Layer1'])  
h_now =  tf.nn.relu(hidden1)

# obtain Q estimate for the next state:
hidden2 = tf.matmul(state_next, w_tgt_1 )   
h_next =  tf.nn.relu(hidden2)

# obtain Q estimate for present and next states:                            
Q_now = tf.matmul(h_now , w['Layer2'])  
Q_next = tf.matmul(h_next, w_tgt_h)
Q_temp = tf.reshape(tf.gather_nd(Q_now,act),[-1,1]) # reshape to find target_Q   
# find the max score for next state - to be used for setting the target Q using greedy policy
target_Q1 = tf.reshape(tf.reduce_max(Q_next,1), [-1,1])  
     
# Update target Q:  
target_Q = (df * tf.stop_gradient(tf.multiply(target_Q1, r + 1))) + r 
# where, r + 1 marks the end of an episode and can be used to calculate 

# find the max scoring action for present state
max_act = tf.arg_max(Q_now, 1)

# find mean square loss 
loss = tf.reduce_mean(0.5*tf.square(target_Q - Q_temp))

# update model parameters using GD 
optimum = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
update = optimum.minimize(loss)

# initialise more variables            
train_ep_len = []
eval_len = [] # no. of moves
eval_r = [] # reward 
eval_loss = [] # for storing mean loss
loss_train = [] # training loss 

#################################--TRAIN--###################################
if Train == True:   
    with tf.Session() as sess: # start trianing
        sess.run(tf.global_variables_initializer())
    
        print("\n***   Starting training   *** \n")
        
        ReBuff = MakeBuffer(start_ep)
    
        for i in range(episodes):
                #  obtain observation (== present state)
                obs = env.reset()
                # reset episode length counter
                cc = 0
                loss_ep = 0 # loss per episode
                # copy the trained network weights 
                if i==0 or i%5==0 : # copy every 5 episodes
                    w_copy1, w_copyh = sess.run([w['Layer1'], w['Layer2']], feed_dict = {})
    
                while cc < (max_len):  # at most 300 moves
                    cc += 1 # add to epsiode length counter
                    obs_temp = obs.reshape([1,4]) # store old opservation in temporary 
                    # define TF feed_dict for e-greedy policy
                    feed = {state_now: obs_temp}
                    pred_act = sess.run(max_act, feed_dict = feed) # fully greedy
                  # agent's action prediction:
                    pred_act = MYgreed(epsilon, pred_act[0]) # modify to e-greedy using helper function
                    
                    # implement predicted action:
                    obs, rew, done, _ = env.step(pred_act)
                    obs = obs.reshape([1,4])
                    # assign reward as per assignment specification
                    if done  & (cc < max_len):
                        rew = -1
                    else:
                        rew = 0
                        
                     # add generated state vector (experience) to the replay buffer:
                    ReBuff.Plus([obs_temp, obs, pred_act, rew, (not done)*1])
                    if done:
                        break
                    
   # trian network using batches
                    # make batches of experience:
                    batch_state, batch_next_state, batch_act, batch_r, batch_done = \
                    MakeBatch2(ReBuff.RandBuff(batch_size))
    
                    # update gradient 
                    action = []
                    for ii in range(batch_size):
                        action.append([ii, batch_act[ii]])
                        
                    feed = {state_now: batch_state, state_next: batch_next_state, act: action, r: batch_r, isdone: batch_done, w_tgt_1: w_copy1, w_tgt_h: w_copyh}
                    up = sess.run(update, feed_dict = feed) #update gradients
                                               
                # Asses learned agent against environment 
                if i%20 == 0:  # evaluate every 20 episodes 
                    loss_mean = 0 
                    sum_ep_len = 0 # episode length
                    sum_r = 0 # return
                    for _ in range(eval_ep_N):
                        loss_ep = 0 # loss per episode
                        cc = 0 #  counter for episode length
                        #  obtain observation (== present state)
                        obs = env.reset()
                        while cc < (max_len):
                            cc += 1 
                            # define dictionary to feed to TF graph
                            feed2 = {state_now: obs.reshape(1,4)}
                            # get action from the thus far learned agent:
                            pred_act = sess.run(max_act, feed_dict = feed2)
                            # originally, action prediction based on a e_greedy policy :
                            # set to purely greedy for evaluation by changing epsilon
                            epsilon = 0 
                            pred_act = MYgreed(epsilon, pred_act[0]) 
                            # implement predicted action:
                            obs, rew, done, _= env.step(pred_act)
                            # assign reward as per assignment specification
                            if done  & (cc < max_len):
                                rew = -1
                                break 
                            else:
                                rew = 0
                                
                            action = [[0,pred_act]]
                            reward = [[rew]]
                            done2 = [[(not done)*1]]
                            # define dictionary to feed to TF graph
                            feed = {state_now: obs_temp.reshape(1,4), state_next: obs.reshape(1,4), act: action, r: reward, isdone: done2, w_tgt_1: w_copy1, w_tgt_h: w_copyh}
                            #  caluclate loss (loss)
                            batch_loss = sess.run(loss, feed_dict = feed)
                            # sum loss over batches into a cumulative episode loss 
                            loss_ep += batch_loss
                            if done:
                                break
                        # aggregate results
                        loss_mean += (loss_ep/ cc)
                        sum_ep_len += cc # cumulative episode length
                        sum_r += (-df ** (cc-1)) # cumulative return            
    
    
                    # calc average episode length and reward 
                    eval_loss.append(loss_mean/ eval_ep_N)
                    eval_len.append(sum_ep_len / eval_ep_N)
                    eval_r.append(sum_r / eval_ep_N)
                    
                    # display results
                    print("episode:", i)
                    idx = int(i/20)
                    print("Mean Loss:", eval_loss[idx-1]) # to check if loss is being calculated
                    print("Mean episode length:", eval_len[idx-1])
                    print("Mean episode return:", eval_r[idx-1])
    
       # Save model
        # if no path than make path
        if not os.path.exists(folder):
                print('Creating path where to save model: ' + folder)
                os.mkdir(folder)
        saver = tf.train.Saver()    
        saver.save(sess,file_save)
        print("\nModel saved at: " + file_save)
         
    #define parameters for plots and reference
    params =  "PartA7, Learning_Rate: " + str(learn_rate)
    # save
    np.savez(file, params = params, eval_loss = eval_loss, eval_len = eval_len, eval_r = eval_r)
    
# LOAD saved results
   
else:
    loaded = np.load(file + '.npz')   
    params = str(loaded['params'])
    eval_loss = loaded['eval_loss']
    eval_len = loaded['eval_len']
    eval_r = loaded['eval_r']
    print('model loaded')

##################################--PLOT--###################################   

# plots of performance :

mtp.figure()
mtp.plot(range(1, episodes, eval_ep_N), eval_len) # define x axis (otherwise scale is 20 times smaller)

mtp.title('Mean episode length vs Index\n' + params, fontsize = 12)
mtp.ylabel('Episode length')
mtp.xlabel('Episode number')

mtp.figure()
mtp.plot(range(1, episodes, eval_ep_N), eval_r) # define x axis (otherwise scale is 20 times smaller)

mtp.title('Mean episode reward vs Index\n' + params, fontsize = 12)
mtp.ylabel('Episode reward')
mtp.xlabel('Episode number')
   
# plot of trianing Loss == learning curve
mtp.figure()
mtp.plot(range(1, episodes, eval_ep_N), eval_loss) # define x axis (otherwise scale is 20 times smaller)
mtp.title('Mean Squared loss vs Index\n' + params, fontsize = 12) 
mtp.xlabel('Episode number')
mtp.ylabel('Mean Loss')