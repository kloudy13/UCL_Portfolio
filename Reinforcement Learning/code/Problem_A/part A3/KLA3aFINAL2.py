# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:54:47 2017

@author: klaudia
Task A: Cart-Pole problem  
Part 3 task:
Collect 2000 episodes under a uniformly random policy and implement 
batch Q-learning to learn to control the cart-pole. To represent the value 
function, use i) a linear transformation and ii) a 1 hidden layer(100) network.
Plot your performance and loss during training. 
Report results for learning rates in [10−5, 0−4, 10−3, 10−2, 10−1, 0.5].

"""
# ============================================================================
# ========================== IMPORTANT !!!! ==================================
# SETUP to save/load
Train =  False # use this setting to load the model
# if Train = True model will train and save results 
# if Train = False model will LOAD and summary results will be displayed

####################### imports ########################
import gym
import numpy as np
import matplotlib.pyplot as mtp
import tensorflow as tf

######################## SETUP ##########################

# frame problem:
max_len = 300 # maximum episode length
df = 0.99 # discount factor
# initialise variables 
episodes = 2000 # number of episodes 
batch_size = 1000 # chosen after trial and error 
learn_rate = [1e-5,1e-4,1e-3,1e-2,1e-1,0.5]  # learning rates to be iterated over
state_size = 4   # size of state vector
poss_action = 2  # number of possible actions
hidden_size = 100  # size of hidden layer
data = [] # initially empty list, to be populated with  2000 episodes generated under a uniformly random policy
epochs = 100 # no. of training epochs 
eval_ep_N = 20# no. of  episodes for evaluation 

# load Cart pole game 
env = gym.make('CartPole-v1') #use this verison as it allows for more than 200 training episodes (unlike v01)
np.set_printoptions(threshold=np.nan)

# set time seeds:
np.random.seed(20)
tf.set_random_seed(20)
env.seed(20) 

######################## FUNCTIONS ##########################

def MakeBatch(data):
#make batches of data coming in the form : [old_observation, observation, action, reward]
    
# create arrays with first data instance written to them 
    batch_state_now = data[0][0] # current batch state
    batch_state_next = data[0][1] # next batch state
    batch_act = data[0][2] # batch action
    batch_r = data[0][3] # batch reward
    # add consecutive data instances to create batch
    for i in range(1,len(data)):
        batch_state_now = np.vstack((batch_state_now ,data[i][0]))
        batch_state_next = np.vstack((batch_state_next ,data[i][1]))
        batch_act = np.vstack((batch_act ,data[i][2]))
        batch_r = np.vstack((batch_r ,data[i][3]))
         
    return batch_state_now, batch_state_next, batch_act, batch_r

######################## LOAD ##########################
if Train == False:
    for l in learn_rate:
        file = "PartA3_(i)_Batch:" + str(batch_size) + "_LR:" + str(l)
        loaded = np.load(file + '.npz')   
        params = str(loaded['params'])
        loss_train = loaded['loss_train']
        eval_len = loaded['eval_len']
        eval_r = loaded['eval_r']
        # plot
        # plots of performance (mean episode length and mean return per episode)
        mtp.figure()
        mtp.plot(range(0,5*len(eval_r),5),eval_r)
        mtp.title('Mean return per episode vs Index \n' + file, fontsize = 12) 
        mtp.ylabel('Mean return')
        mtp.xlabel('Epoch number')
        
        mtp.figure()
        mtp.plot(range(0,5*len(eval_len),5), eval_len)
        mtp.axis([0, epochs, 0, max_len])
        mtp.title('Mean episode length vs Index\n' + file, fontsize = 12)
        mtp.ylabel('Episode length')
        mtp.xlabel('Epoch number')
       
       # plot of trianing Loss
        mtp.figure()
        mtp.plot(loss_train) 
        mtp.title('Mean treainig batch loss vs Batch number\n' + file, fontsize = 12) 
        mtp.xlabel('Batch number')
        mtp.ylabel('Loss')
        
else:
##################################--MAIN--###################################
    
    # generate data for 2000 random episodes
    for _ in range(episodes):
        done = False 
        obs = env.reset() # initialise observation
        c = 0 # initialise counter
      
        while c < (max_len)-1:
            old_obs = obs # keep old observation             
            # sample random action from uniform distrib. (uniform random policy)
            act = 1*(np.random.rand()>0.5)       
            #  update 
            obs, r, done, _ = env.step(act) # obs is observation, r is reward
            c += 1 # update counter
            if done:
                r = -1 # set final reward
                data.append([old_obs, obs, act, r])
                break 
            else:
                r = 0 # reward is 0 on non-terminating steps
                data.append([old_obs, obs, act, r]) 
    
    # loop over the various learning rates
    for l in learn_rate:
        
        tf.reset_default_graph() # reset graph 
        
        # initialise placeholders for state vectors:
        state_now = tf.placeholder(tf.float32, shape = [None,state_size], name ='state_now')
        state_next = tf.placeholder(tf.float32, shape = [None,state_size], name ='state_next')
        # initialise placeholder for actions:                
        act = tf.placeholder(tf.int32, shape = [None,poss_action], name ='action')
        # initialise placeholder for reward:                      
        r = tf.placeholder(tf.float32, shape = [None,1], name ='reward')
        
    # Define NN and Weights
        # weights initialiser:
        init = tf.contrib.layers.xavier_initializer(uniform=True, seed = 20, dtype=tf.float32)
        
        w = {
            'Layer1': tf.get_variable('w1', shape=[state_size, poss_action], initializer = init)} 
        
        # get Q estimate for present and next states:                            
        Q_now = tf.matmul(state_now , w['Layer1'])  
        Q_next = tf.matmul(state_next, w['Layer1'])
        Q_temp = tf.reshape(tf.gather_nd(Q_now,act),[-1,1]) # reshape to find target_Q
        
        # find the max score for next state - to be used for setting the target Q using greedy policy
        target_Q1 = tf.reshape(tf.reduce_max(Q_next,1), [-1,1])  
             
        # Update target Q:  
        target_Q = (df * tf.stop_gradient(tf.multiply(target_Q1, r + 1))) + r 
        # where, r + 1 marks the end of an episode and can be used to calculate 
        # expected reward at that tilmestep 
        difference = target_Q - Q_temp
        
        # find the max scoring action for present state
        max_act = tf.arg_max(Q_now, 1)
        
        # find mean square loss 
        loss = tf.reduce_mean(0.5*tf.square(difference))
        
        # update model parameters using RMSPropOptimizer
        optimum = tf.train.RMSPropOptimizer(learning_rate = l)
        update = optimum.minimize(loss)
               
        # initialise more variables
        eval_len = [] # no. of moves
        eval_r = [] # reward 
        loss_train = [] # training loss 
    
    ##################################-TRAIN--###################################
        
        # start trianing
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            print("\n***   Starting training with learning rate: ", l, "  *** \n")
            
            for i in range(epochs):
                np.random.shuffle(data)
                for b in range(0,(len(data)//batch_size)):
                    # make batches of data using helper function:
                    batch_state, batch_next_state, batch_act, batch_r = MakeBatch(data[batch_size * b : batch_size * (b+1)][:])
    
                    # update gradient and calculate loss:
                    action = []
                    for ii in range(len(batch_act)):
                        np.array(action.append([ii, batch_act[ii]]))
                        
                    feed = {state_now: batch_state, state_next: batch_next_state, act: action, r: batch_r}
                    _, batch_loss = sess.run([update, loss], feed_dict = feed)
                                    
                    loss_train.append(batch_loss) # record training loss in list
    
                # Asses learned agent against environment 
                if i%5 == 0:     # evaluate every 5 counters
                    sum_ep_len = 0 # episode length
                    sum_r = 0 # return
                    for _ in range(eval_ep_N):
                        #  obtain observation (== present state)
                        obs = env.reset()
                        cc = 0 #  counter for episode length
                        while cc < max_len - 1:
                            cc += 1                 
                            # get action from the thus far learned agent:
                            feed_dict = {state_now: obs.reshape(1,4)}
                            # action prediction based on a greedy policy: (picking max value)
                            pred_act = sess.run(max_act, feed_dict = feed_dict)
                            # implement predicted action:
                            obs, rew, done, _ = env.step(int(pred_act))
                            # assign reward as per assignment specification
                            if done:
                                rew = -1
                                break 
                            else:
                                rew = 0
                        sum_ep_len += cc # cumulative episode length
                        sum_r += (-df ** (cc-1)) # cumulative return
                   
                    # calc average episode length and reward 
                    eval_len.append(sum_ep_len / eval_ep_N)
                    eval_r.append(sum_r / eval_ep_N)
                    
                    # display results
                    print("epoch: ", i)
                    print("Loss: ", loss_train[i-1]) # to check if loss is being calculated
                    print("Mean episode length: ", (sum_ep_len / eval_ep_N))
    
        #define parameters for plots and reference
        params = "Batch_Size: " + str(batch_size) + ", Learning_Rate: " + str(l)
        
        # SAVE
        file = "PartA3_(i)_Batch:" + str(batch_size) + "_LR:" + str(l)
        np.savez(file, params = params, loss_train = loss_train, eval_len = eval_len, eval_r = eval_r)
                 
    ##################################--PLOT--###################################
    
    # plots of performance (mean episode length and mean return per episode)
        mtp.figure()
        mtp.plot(range(0,5*len(eval_r),5),eval_r)
        mtp.title('Mean return per episode vs Index \n' + file, fontsize = 12) 
        mtp.ylabel('Mean return')
        mtp.xlabel('Epoch number')
        
        mtp.figure()
        mtp.plot(range(0,5*len(eval_len),5), eval_len)
        mtp.axis([0, epochs, 0, max_len])
        mtp.title('Mean episode length vs Index\n' + file, fontsize = 12)
        mtp.ylabel('Episode length')
        mtp.xlabel('Epoch number')
       
    # plot of trianing Loss
        mtp.figure()
        mtp.plot(loss_train) 
        mtp.title('Mean treainig batch loss vs Batch number\n' + file, fontsize = 12) 
        mtp.xlabel('Batch number')
        mtp.ylabel('Loss')
    
