# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 10:20:15 2017

@author: klaudia
Task A: Cart-Pole problem  
Part 5 task:
Make a smaller network using only 30 unit in the hidden layer with ReLUs. 
Now try more, hidden layer with 1000 hidden units. 
Compare and report their performance over 2000 episodes.
(Plot the test performance and bellman loss every 20 steps).
 
Will tackle in two parts: part a) will be the small hidden unit and part b) the large one

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
import os.path

######################## SETUP ##########################

# frame problem:
max_len = 300 # maximum episode length
df = 0.99 # discount factor
# initialise variables 
episodes = 2000 # number of episodes 
#batch_size = 1000 # chosen after trial and error 
learn_rate = 0.1  # learning rates to be iterated over
state_size = 4   # size of state vector
poss_action = 2  # number of possible actions
hidden_size = 1000  # size of hidden layer
epsilon = 0.05 # set constant for e_greedy policy
eval_ep_N = 20 # no. of  episodes for evaluation 

# for load and save:
folder = "models/"
file = "PartA5b_ "+"LR:" + str(learn_rate)
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
    
##################################--MAIN--###################################
                      
tf.reset_default_graph() # reset graph 

# initialise placeholders for state vectors:
state_now = tf.placeholder(tf.float32, shape = [None,state_size], name ='state_now')
state_next = tf.placeholder(tf.float32, shape = [None,state_size], name ='state_next')
# initialise placeholder for actions:                
act = tf.placeholder(tf.int32, shape = [None,poss_action], name ='action')
# initialise placeholder for reward:                      
r = tf.placeholder(tf.float32, shape = [None,1], name ='reward')

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
hidden2 = tf.matmul(state_next, w['Layer1'])   
h_next =  tf.nn.relu(hidden2)

# obtain Q estimate for present and next states:                            
Q_now = tf.matmul(h_now , w['Layer2'])  
Q_next = tf.matmul(h_next, w['Layer2'])
Q_temp = tf.reshape(tf.gather_nd(Q_now, act),[-1,1]) # reshape to find target_Q   
   # find the max score for next state - to be used for setting the target Q using greedy policy
target_Q1 = tf.reshape(tf.reduce_max(Q_next, 1), [-1,1])  

# Update target Q:  
target_Q = (df * tf.stop_gradient(tf.multiply(target_Q1, r + 1))) + r 
# where, r + 1 marks the end of an episode and can be used to calculate 
# expected reward at that tilmestep
 
# find the max scoring action for present state
max_act = tf.arg_max(Q_now, 1)

# find mean square loss 
loss = tf.reduce_mean(0.5*tf.square(target_Q - Q_temp))

# update model parameters using GD 
optimum = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
update = optimum.minimize(loss)
       
# initialise more variables for training evaluation          
eval_len = [] # no. of moves
eval_r = [] # reward 
eval_loss = [] # for storing mean loss

#################################--TRAIN--###################################
if Train == True:   
    with tf.Session() as sess: # start trianing
        sess.run(tf.global_variables_initializer())
    
        print("\n***   Starting training   *** \n")
        
        for i in range(1, episodes+1):
                #  obtain observation (== present state)
                obs = env.reset()
                # reset episode length counter
                cc = 0
                loss_ep = 0 # loss per episode
                
                while cc < (max_len): # at most 300 moves
                    cc += 1 # add to epsiode length counter
                    obs_temp = obs # store old opservation in temporary 
                    # define TF feed_dict for e-greedy policy
                    feed = {state_now: obs_temp.reshape(1,4)}
                    pred_act = sess.run(max_act, feed_dict = feed) # fully greedy
                    # originally, action prediction based on a e_greedy policy :
                    # set to purely greedy for evaluation by changing epsilon
                    epsilon = 0
                    pred_act = MYgreed(epsilon, pred_act[0]) 
                    # implement predicted action:
                    obs, rew, done, _ = env.step(pred_act)
                    # assign reward as per assignment specification
                    if done & (cc < max_len):
                        rew = -1
                    else:
                        rew = 0
                    
                    action = [[0,pred_act]]
                    reward = [[rew]]
                    # define dictionary to feed to TF graph
                    feed = {state_now: obs_temp.reshape(1,4),act: action,r: reward, state_next: obs.reshape(1,4)}
                    # update gradients (update) and caluclate loss (loss)
                    _, _= sess.run([update, loss], feed_dict = feed)
                    if done:
                        break
                    
 # EVALUATE               
                # Asses learned agent against environment 
                if i%20 == 0 or i==0 :  # evaluate every 20 episodes 
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
                            if done & (cc < max_len):
                                rew = -1
                                break 
                            else:
                                rew = 0
                            action = [[0,pred_act]]
                            reward = [[rew]]
                            # define dictionary to feed to TF graph
                            feed = {state_now: obs_temp.reshape(1,4),act: action, r: reward, state_next: obs.reshape(1,4)}
                            # update gradients (update) and caluclate loss (loss)
                            batch_loss = sess.run(loss, feed_dict = feed)
                            # sum loss over batches into a cumulative episode loss 
                            loss_ep += batch_loss
                            if done:
                                break
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
    params =  "PartA5b, Learning_Rate: " + str(learn_rate) + 'Hidden Layer size: '+str(hidden_size)
    
    # save selected variables 
    np.savez(file, params = params, eval_loss = eval_loss, eval_len = eval_len, eval_r = eval_r)

else: #load   
    loaded = np.load(file + '.npz')   
    params = str(loaded['params'])
    eval_loss = loaded['eval_loss']
    eval_len = loaded['eval_len']
    eval_r = loaded['eval_r']
    print('model loaded')
    
##################################--PLOT--###################################   
   
mtp.figure()
mtp.plot(eval_len)
mtp.title('Mean episode length vs Index\n' + params, fontsize = 12)
mtp.ylabel('Episode length')
mtp.xlabel('Episode number')

mtp.figure()
mtp.plot(eval_r)
mtp.title('Mean episode reward vs Index\n' + params, fontsize = 12)
mtp.ylabel('Episode reward')
mtp.xlabel('Episode number')
   
# plot of trianing Loss == learning curve
mtp.figure()
mtp.plot(eval_loss) 
mtp.title('Mean Squared loss vs Index\n' + params, fontsize = 12) 
mtp.xlabel('Episode number')
mtp.ylabel('Mean Loss')
   
   


