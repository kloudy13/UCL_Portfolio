# -*- coding: utf-8 -*-
"""
@author: klaudia
Problem B atari games 
parts 3 and 4
"""

# ============================================================================
# ========================== IMPORTANT !!!! ==================================

# SETUP to save/load
Train = False # use this setting to load the model
# if Train = True model will train and save results 
# if Train = False model will LOAD and summary results will be displayed

# CHOSE GAME 
# To load relevant model chose between environment name: 'Box'/ 'Pacman'/ 'Pong'
env_name = 'Box' #'Pong' # 'Pacman'

####################### imports ########################
import gym
import numpy as np
import matplotlib.pyplot as mtp
import scipy.misc
import tensorflow as tf
import os.path
import random

######################## SETUP ##########################
# frame problem:
max_len = 300 # maximum episode length
df = 0.99 # discount factor
# initialise variables 
step_len = 1000000 # train for up to a million agent steps
eval_run_size = 50000 #evaluate your agent every 50 000 steps
img_dim = 28 # image height or width
state_size = img_dim * img_dim  # size of state vector
batch_size = 32 # minibatch size 
learn_rate = 0.001 # learning rate
eval_ep_N = 20                 
epsilon = 0.1 # set constant for e_greedy policy 
hidden_size = 256  # size of hidden layer as per assignment spec
flat_size = 7*7*32 # convolutional layer output shape (32 is size of 2nd filter)
buff_size = 110000   # size of the replay buffer 
start_ep = 100  # number of start episodes to be used
# RMS optimiser parameters:
#decay = 0.9  # decay constant                                             
#eps = 0.1 # epsilon for RMS      
# these were treid but no improvemnet so in the end default values used
                                       
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
file = "PartB_3&4"+env_name+"_LR:" + str(learn_rate)
file_save = folder + file
file_N = file + '.npz' # for numpy save/load variables
load_eval_steps = 100 # evalutaion length when loading model 

# set time seeds:
np.random.seed(20)
tf.set_random_seed(20)
env.seed(20) 

############################### FUNCTIONS #################################

def MYgreed(epsilon, q):
    test = np.random.rand()
    if (test <= epsilon):
        return env.action_space.sample()
    else:
        return q  

def MakeBatch(data,state_size):
#make batches of data coming in the form : [old_observation, observation, action, reward, done]
    # create empty arrays  
    batch_state_now = np.zeros([batch_size,state_size,4]) # current batch state
    batch_state_next = np.zeros([batch_size,state_size,4]) # next batch state
    batch_act = np.zeros([batch_size,1]) # batch action
    batch_r =  np.zeros([batch_size,1]) # batch reward
    batch_done =  np.zeros([batch_size,1])
    # add consecutive data instances to create batch
    for i in range(1,len(data)):
        batch_state_now[i,] = data[i][0]
        batch_act[i,] = data[i][1]
        batch_r[i,] = data[i][2]
        batch_state_next[i,] = data[i][3]
        batch_done[i,] = data[i][4]
#    for i in range(1,len(data)):
#        batch_state_now = np.vstack((batch_state_now ,data[i][0]))
#        batch_act = np.vstack((batch_act ,data[i][1]))
#        batch_r = np.vstack((batch_r ,data[i][2]))
#        batch_state_next = np.vstack((batch_state_next ,data[i][3]))
#        batch_done = np.vstack((batch_done, data[i][4]))
    return batch_state_now, batch_act, batch_r, batch_state_next, batch_done

def Update1(obs, buff):
# returns a new state buffer
# where oldest observation is dicarded and replaced by a new state
    temp = buff
    new_state = np.zeros_like(buff)
    new_state[:,0] = obs
    new_state[:,1] = temp[:,0]
    new_state[:,2] = temp[:,1]
    new_state[:,3] = temp[:,2] 
    return new_state

def MakeBuffer2(episodes): #check
# function to make random uniform data for the episodes   
    data = RunBuffer()
    for i_episode in range(episodes):
        done = False
        obs = env.reset() # initialise observation
        obs = Grayscale(obs) # reshape and turn to grayscale
        act = env.action_space.sample() # get intial action
        # initailise buffers
        buffer1 = np.ones([state_size,4])
        buffer2 = buffer1 
        for _ in range(4):
            buffer1 =  Update1(obs, buffer1)
        while not done:
            obs, rew, done, _ = env.step(act)
            obs = Grayscale(obs) # reshape and turn to grayscale
            rew = np.clip(rew, -1, 1)
            buffer2 = Update1(obs, buffer1)
            if done:
                data.Plus([buffer1, act, rew, buffer2, 0.0])
                break 
            data.Plus([buffer1, act, rew, buffer2, (not done)*1.])
            buffer1 = buffer2
            act = env.action_space.sample()
    return data
            

class RunBuffer(object): # check
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
        

def Reshape(data, end_width = 28, end_height = 28):
# reduce data(image) to requred dimensions (end width/height) ste by default to 28 by 28   
    #end_width, end_height = 28, 28
    new_img = scipy.misc.imresize(data, (end_height, end_width))
    return new_img

def Grayscale(data, end_width = 28, end_height = 28):
    new_img = Reshape(data, end_width, end_height)
    # spit into image Red-Green-Blue parts
    Red = new_img[:, :, 0]
    Green = new_img[:, :, 1]
    Blue = new_img[:, :, 2]
    # transform to grayscale:
    gray_img = (Red * 299. / 1000) + (Green * 587. / 1000) + (Blue * 114. / 1000)
    return gray_img.flatten().astype(np.uint8)  #flatten to one array

def Discount(df, rewards):
# function calculates dicoutned rewards 
# inputs : list of rewards and the discount factor (df)
    c = 0.0 
    rewards = np.array(rewards) #change list to array for ease of manipulation
    discounted = np.zeros(rewards.shape) # init. empty array for discounted reward to be written   
    for ind in reversed(range(rewards.size)):
        c = rewards[ind] + c * df  
        discounted[ind] = c
    return discounted[0]
    
##################################--MAIN--###################################

tf.reset_default_graph() # Reset graph

# MAKE PLACEHOLDERS
# initialise placeholders for state vectors:
state_now = tf.placeholder(tf.float32, shape=[None,state_size,4], name = 'state_now')
state_next = tf.placeholder(tf.float32, shape=[None,state_size,4],name = 'state_next')
# initialise placeholder for actions:                
act = tf.placeholder(tf.int32, shape=[None,2],name ='action')
# initialise placeholder for reward:                      
r = tf.placeholder(tf.float32, shape=[None,1],name ='reward')
# need to make the 'done' variable into tensor so as to use the expeinece buffer
isdone = tf.placeholder(tf.float32, shape=[None,1], name = 'isdone')
# target weights 1st layer
w_tgt_1 = tf.placeholder(tf.float32, shape=[flat_size, hidden_size], name = 'weight1')  
# target weights hidden layer                            
w_tgt_h = tf.placeholder(tf.float32, shape=[hidden_size, poss_act],name = 'weight_h') 
# define placeholders for conv net weights and biases:
w_c1 = tf.placeholder(tf.float32, shape=[6,6,4,16])  
w_c2 = tf.placeholder(tf.float32, shape=[4,4,16,32])  
b_c1 = tf.placeholder(tf.float32, shape=[1,16]) 
b_c2 = tf.placeholder(tf.float32, shape=[1,32]) 

### Define weights for NN 
# weights initialiser:
init = tf.contrib.layers.xavier_initializer(uniform=True, seed = 20, dtype=tf.float32)

w = {
    'Layer1': tf.get_variable('w1', shape=[flat_size, hidden_size], initializer = init), 
    'Layer2': tf.get_variable('w2', shape=[hidden_size, poss_act], initializer = init),
    'Conv1':  tf.get_variable('wc1', shape=[6,6,4,16], initializer = init),
    'Conv2': tf.get_variable('wc2', shape=[4,4,16,32], initializer = init)}                                               
# add bias terms for conv layers:
b = {
    'Conv1': tf.get_variable('bc1', shape=[1,16], initializer = init),
    'Conv2': tf.get_variable('bc2', shape=[1,32], initializer = init)} 

## BUILD NET 1

# Conv layer 1 with ReLU:
hidd1_c1 = tf.nn.conv2d(tf.reshape(state_now,[-1,img_dim,img_dim,4]), w['Conv1'], strides= [1,2,2,1], padding='SAME')
hidd1_r1 = tf.nn.relu(tf.add(hidd1_c1, b['Conv1']))

# Conv layer 2 with ReLU:
hidd1_c2 = tf.nn.conv2d(hidd1_r1, w['Conv2'], strides= [1,2,2,1], padding='SAME')
hidd1_r2 = tf.nn.relu(tf.add(hidd1_c2,b['Conv2']))

hidd1_flat = tf.reshape(hidd1_r1, [-1, flat_size]) # flatten by 7*7*32
hidd1 = tf.matmul(hidd1_flat, w['Layer1'])   # Q network run
hidd1_now =  tf.nn.relu(hidd1)

## BUILD NET 2 - target 

# Conv layer 1 with ReLU:
hidd2_c1 = tf.nn.conv2d(tf.reshape(state_next,[-1,img_dim,img_dim,4]), w_c1, strides= [1,2,2,1], padding='SAME')
hidd2_r1 = tf.nn.relu(tf.add(hidd2_c1, b_c1))

# Conv layer 2 with ReLU:
hidd2_c2 = tf.nn.conv2d(hidd2_r1, w_c2, strides= [1,2,2,1], padding='SAME')
hidd1_r2 = tf.nn.relu(tf.add(hidd1_c2, b_c2))

ht2_flat = tf.reshape(hidd1_r2, [-1, flat_size])  # flatten by 7*7*32 
hidd2 = tf.matmul(ht2_flat,w_tgt_1)  # Q network run
hidd2_now =  tf.nn.relu(hidd2)

# VALUE FUNCTION
Q_now = tf.matmul(hidd1_now,w['Layer2'])  
Q_next = tf.matmul(hidd2_now, w_tgt_h)
Q_temp = tf.reshape(tf.gather_nd(Q_now, act),[-1,1])

# find the max score for next state - to be used for setting the target Q using greedy policy
target_Q1 = tf.reshape(tf.reduce_max(Q_next,1),[-1,1])   
  
# Update Qtarget based on prediction and current reward
target_Q = r + df * tf.stop_gradient(tf.multiply(isdone, target_Q1)) 

# find the max scoring action for present state
max_act = tf.arg_max(Q_now, 1)

# calculate squared loss (target - predicted )
loss = tf.reduce_mean(0.5*tf.square(target_Q - Q_temp))

# update model unisg RMSPropOptimizer and adjust parameters as per assignment spec.
optimum = tf.train.RMSPropOptimizer(learning_rate = learn_rate)# ,decay = decay, epsilon = eps)
update = optimum.minimize(loss)

#################################--TRAIN--###################################

# initialise more variables
# train
train_loss = [] #list to store training loss
train_ep_len = [] #list to store epsiode length
train_ep_r = [] #list to store rewards
# training evaluate
train_eval_loss = [] #training evaluate loss                
train_eval_r = []  #training evaluate epsiode reward               
train_eval_len = []   #training evaluate epsiode len 

with tf.Session() as sess:
    if Train == True:
        
        sess.run(tf.global_variables_initializer())
        ReBuff = MakeBuffer2(start_ep) #  start by partialy adding start_episodes to the buffer
        print("\n***   Starting training   *** \n")
        j = 0 # init global steps counter 
        
        while (j <= step_len) : # run for 1,000,000 agent steps
        #setup
            done = False
            isEvaluation = False  #create logical statemnet to control when performance evaluation occurs
            loss_ep = 0
            cc = 1  # init local counter
            
            if j%5000 == 0 or j == 0: # compu network every 5000 episodes
            # run graph on weights 
                train_w_c1, train_w_c2, train_b_1, train_b_2, train_w_1, train_w_h =\
                sess.run([w['Conv1'],w['Conv2'],b['Conv1'],b['Conv2'],w['Layer1'], w['Layer2']], feed_dict={})
            
            r_temp = [] # temp storage for reward varaible
            buff1 = np.ones([state_size,4]) #  init. buffer to store 4 images at a time
            buff2 = buff1    # init. future state buffer1        
            obs = env.reset() ## reset environ.
            obs = Grayscale(obs,img_dim,img_dim) # transform observation to desired shape and grayscale               
            for _ in range(4):
                buff1 = Update1(obs, buff1) # update buffer with observation
            action = env.action_space.sample() # init. action
            while not done:
                obs, rew, done, _ = env.step(action)
                obs = Grayscale(obs,img_dim,img_dim)  
                rew = np.clip(rew, -1, 1)
                r_temp.append(rew)
                buff2 = Update1(obs, buff1) # update obervations in buffer2
                if done:
                    ReBuff.Plus([buff1, action, rew, buff2, 0.0])
                    break 
                ReBuff.Plus([buff1, action, rew, buff2, (not done)*1.])
                buff1 = buff2
                # while not done sample another action 
                action = sess.run(max_act, feed_dict={state_now: [buff1]})
                action = MYgreed(epsilon, action[0])
                act_temp = [] # init. list for state actions
                # generate batch 
                batch_state_now, batch_act, batch_r, batch_state_next,batch_done = \
                MakeBatch(ReBuff.RandBuff(batch_size),state_size) # batchtaken from the buffer
                # populate state of action
                for ii in range(batch_size):
                    act_temp.append([ii,batch_act[ii]])
                act_temp=np.asarray(act_temp)
                # define dictionary to feed to TF graph
                feed1 = {state_now: batch_state_now, act: act_temp,r: batch_r, state_next: batch_state_next,isdone: batch_done, w_c1: train_w_c1, w_c2: train_w_c2, b_c1: train_b_1,b_c2: train_b_2, w_tgt_1: train_w_1, w_tgt_h: train_w_h}
                # Update gradient and calc. loss
                up_loss,_ = sess.run([loss,update], feed_dict = feed1)
                loss_ep += up_loss 
                j +=1    # add to counters         
                cc +=1                 
                # as per assignment spec. evaluate every 50000 runs
                if j%eval_run_size == 0: 
                    isEvaluation =True
                    
            # record all calculated parameters and loss 
            train_loss.append(loss_ep/cc)
            train_ep_len.append(cc-1)
            disc_r = Discount(df, r_temp) # discount the reward
            train_ep_r.append(disc_r)
           
            # print some results ever so often
            if j%10==0:
                print("step: ",j, ", loss:",(loss_ep/cc), ", length:",cc)

## EVALUATION           
# as per assignment spec. evaluate every 50000 runs
            if isEvaluation == True:
                print('Training Evaluation in progress, another 50000 runs...')
                # init. more counter variables:
                loss_mean = 0
                sum_ep_len = 0 # episode length
                sum_r = 0 # return
                for _ in range(eval_ep_N):
                    # init. more variables:
                    done = False
                    loss_ep = 0
                    cceval = 1 # init evaluation epsiode counter
                    r_temp = [] # temp storage for reward varaible
                    buff1 = np.ones([state_size,4]) # buffer to store 4 images at a time
                    buff2 = buff1    # future state buffer
                    obs = env.reset() ## reset environ/ init. present observation
                    obs = Grayscale(obs,img_dim,img_dim)  # transform observation to desired shape and grayscale                             
                    for _ in range(4): # write 1st observation to  buffer
                        buff1 = Update1(obs, buff1)
                    # init. new action
                    action = env.action_space.sample()
                    while not done:
                        obs, rew, done, _ = env.step(action)
                        obs = Grayscale(obs)  
                        rew = np.clip(rew, -1, 1)
                        r_temp.append(rew)
                        # update obervations in buffer
                        buff2 = Update1(obs, buff1) 
                        ReBuff.Plus([buff1, action, rew, buff2, (not done)*1.])
                        # define dictionary to feed to TF graph
                        feed = {state_now: np.reshape(buff1,[1,state_size,4]), act:np.reshape([0,action],[1,2]), r:np.reshape(rew,[1,1]), state_next:np.reshape(buff2,[1,state_size,4]),isdone:np.reshape(done,[1,1]), w_c1: train_w_c1, w_c2: train_w_c2, b_c1: train_b_1,b_c2: train_b_2, w_tgt_1: train_w_1, w_tgt_h: train_w_h}
                        # Update gradient and calc. loss
                        _,up_loss = sess.run([update, loss], feed_dict = feed)
                        if done:
                            ReBuff.Plus([buff1, action, rew, buff2, 0.0])
                            break 
                        buff1 = buff2 #update buffer
                        # while not done sample another action 
                        action = sess.run(max_act, feed_dict={state_now: [buff1]})
                        action = MYgreed(0.0, action[0])
                        cceval +=1 # add to counter
                        loss_ep += up_loss # sum training loss into episode loss
                    sum_ep_len += cceval
                    disc_r = Discount(df, r_temp) # discount the reward
                    sum_r += disc_r
                    loss_mean += (loss_ep/cceval)
                 
                # calc average episode length, reward and loss:
                train_eval_loss.append(loss_mean/eval_ep_N)
                train_eval_r.append(sum_r/eval_ep_N)
                train_eval_len.append(sum_ep_len/eval_ep_N)
                
                isEvaluation = False # end training evaluation             

### Save Trained Model
        # if no path than make path
        if not os.path.exists(folder):
                print('Creating path where to save model: ' + folder)
                os.mkdir(folder)
        saver = tf.train.Saver()    
        saver.save(sess, file_save)  
        print("\nModel saved at: " + file_save)
        
        # average results 
        mean_train_ep_len = np.mean(train_ep_len)   
        mean_train_ep_r = np.mean(train_ep_r)
        
        # display results    
        print("\nTraining average results over a 100 episodes:\n")
        print("Mean episode length:", mean_train_ep_len )
        print("Mean return from initial state:", mean_train_ep_r )
        
       #define parameters for plots and reference
        params =  "PartB. "+ env_name + "_Learning_Rate: " + str(learn_rate)

        # Save results    
        np.savez(file_N, params = params,train_ep_len = train_ep_len, train_ep_r = train_ep_r, train_eval_loss = train_eval_loss, train_eval_r = train_eval_r, train_loss = train_loss, train_eval_len = train_eval_len, train_w_c1= train_w_c1, train_w_c2 = train_w_c2, train_b_1 = train_b_1, train_b_2 = train_b_2, train_w_1 = train_w_1, train_w_h = train_w_h)
              
############################--LOADING--############################
      # load saved model and evaluate over 100 epsiodes               
    else: 
        sess.run(tf.global_variables_initializer())
        
        # initialise some variables after loading  (to evaluate loaded model) 
        load_eval_loss = []  
        load_eval_len = []
        load_eval_r = []
        
        print("\nLoading saved model: "+ file_save)
        # load saved session
        loader = tf.train.Saver()
        loader.restore(sess, file_save)
        print("Model Loaded\n") 
        # load additional saved variables 
        loaded = np.load(file_N)
        params = str(loaded['params'])
        train_ep_len = loaded['train_ep_len']
        train_ep_r = loaded['train_ep_r']
        train_loss = loaded['train_loss']   
        train_eval_len = loaded['train_eval_len']
        train_eval_loss = loaded['train_eval_loss']
        train_eval_r = loaded['train_eval_r']  
        #load trained model weights
        train_w_c1 = loaded['train_w_c1']
        train_w_c2 = loaded['train_w_c2']
        train_b_1 = loaded['train_b_1']
        train_b_2 = loaded['train_b_2']
        train_w_1 = loaded['train_w_1']
        train_w_h = loaded['train_w_h']
        
        print("Starting performance evaluation\n")
    
########################-- Evaluate-LOADED-MODEL--###########################
        loss_mean = 0
        sum_ep_len = 0 # episode length
        sum_r = 0 # return
        for _ in range(0,load_eval_steps): 
            done = False
            cceval = 1 
            loss_ep = 0
            r_temp = [] # temp storage for reward varaible
            buff1 = np.ones([state_size,4]) #  init. buffer to store 4 images at a time
            buff2 = buff1    # init. future state buffer   
            obs = env.reset() ## reset environ /init. observation
            obs = Grayscale(obs) # transform observation to desired shape and grayscale                     
            for _ in range(4):  # write 1st observation to  buffer
                buff1= Update1(obs, buff1) 
            # sample first action
            action = env.action_space.sample() # init. action
            while not done: 
                # recalc. environment based on action
                obs, rew, done, _ = env.step(action)
                obs = Grayscale(obs)  
                rew = np.clip(rew, -1, 1) #clip reward
                r_temp.append(rew)
                # update obervations in buffer
                buff2 = Update1(obs, buff1)
                # define dictionary to feed to TF graph
                feed1 = {state_now: np.reshape(buff1,[1,state_size,4]), act:np.reshape([0,action],[1,2]), r:np.reshape(rew,[1,1]), state_next:np.reshape(buff2,[1,state_size,4]), isdone:np.reshape(done,[1,1]), w_c1: train_w_c1, w_c2: train_w_c2, b_c1: train_b_1,b_c2: train_b_2, w_tgt_1: train_w_1, w_tgt_h: train_w_h}
                # update gradient and get loss
                _, up_loss = sess.run([update, loss], feed_dict = feed1)
                if done:
                    break 
                # get action from the thus far learned agent:
                buff1 = buff2
                action = sess.run(max_act, feed_dict={state_now: [buff1]})
                action = MYgreed(0.0, action[0])
                # sum loss into a cumulative episode loss 
                loss_ep += up_loss
                cceval+=1 # add to counter
           
            loss_mean += (loss_ep/cceval)
            sum_ep_len += cceval
            disc_r = Discount(df, r_temp) # discount the reward
            sum_r += disc_r
            # calc average episode length, reward and append to relevant list together with loss
            load_eval_loss.append((loss_ep/cceval))
            load_eval_len.append(cceval)
            load_eval_r.append(disc_r) 
            
        #####
        # calc resulst for part 4:
        # Report the final performance in terms of cumulative discounted rewards 
        # per episode of your trained agent, averaged over 100 episodes
        mean_eval_len = np.mean(load_eval_len)   
        mean_eval_r = np.mean(load_eval_r)
        
        # display results    
        print("Part 4. Performance evaluation results averaged over a 100 episodes:\n")
        print("Mean episode length:", mean_eval_len )
        print("Mean return from initial state:", mean_eval_r )
        

##########################--TRAINING-PLOTS-PART-3--#######################

#scale axis to go back to display evaluation resulst per step
axis_adjust = len(train_eval_loss)
r =range(0,axis_adjust * eval_run_size, eval_run_size)

print('Training Model Evaluation every 50,000 steps\n')
# plot evaluation training loss
mtp.figure()
mtp.plot(r,train_eval_loss) 
mtp.suptitle('Evaluaton training loss per episode vs Index\n'+ params, fontsize = 12)
mtp.xlabel('Step')
mtp.ylabel('Ave. Bellman Loss')

# plot PERFORMANCE (length and reward):
# average discounted rewards   
mtp.figure()
mtp.plot(r, train_eval_r)
mtp.title('Evaluaton training discounted reward vs Index\n'+ params, fontsize = 12)
mtp.xlabel(' Step ')
mtp.ylabel('Ave. Discounted Reward')

# average episode length
mtp.figure()
mtp.plot(r, train_eval_len)
mtp.title('Evaluaton training episode length vs Index\n'+ params, fontsize = 12)
mtp.xlabel('Step')
mtp.ylabel('Ave. Episode Length')


############################--LOADED-RESULTS-PART-4--########################
if Train == False:
    print('Loaded Model Evaluation for PART 4 --> TEST TIME!')
# plot LOSS
    mtp.figure()
    mtp.plot(load_eval_loss) 
    mtp.title('Test loss per episode vs Index\n' + params, fontsize = 12)
    mtp.xlabel('Episode number')
    mtp.ylabel('Ave. Bellman Loss')
    
# plot PERFORMANCE (length and reward):
    # average discounted rewards   
    mtp.figure()
    mtp.plot(load_eval_r)
    mtp.title('Test discounted reward vs Index\n'+ params, fontsize = 12)
    mtp.xlabel('Episode number')
    mtp.ylabel('Ave. Discounted Reward')

    # average episode length
    mtp.figure()
    mtp.plot(load_eval_len)
    mtp.title('Test episode length vs Index\n'+ params, fontsize = 12)
    mtp.xlabel('Episode number')
    mtp.ylabel('Ave. Episode Length')
   

