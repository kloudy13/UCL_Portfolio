# -*- coding: utf-8 -*-
"""
@author: klaudia
Problem B atari games 
Part 2:
Report performance on the three games from an initialised but untrained
Q-network, evaluated on 100 episodes. Explain why the performance can be
different from part one.
"""

# ============================================================================
# ========================== IMPORTANT !!!! ==================================
# SETUP to save/load
Train = False # use this setting to load the model
# if Train = True model will train and save results 
# if Train = False summary results will be displayed

# CHOSE GAME 
# To load relevant model chose between environment name: 'Box'/ 'Pacman'/ 'Pong'
env_name = 'Box'# 'Pacman'

####################### imports ########################
import gym
import numpy as np
import tensorflow as tf
import scipy.misc
import os.path

######################## SETUP ##########################

# frame problem:
df = 0.99 # discount factor
# initialise variables 
img_dim = 28 # image height or width
state_size = img_dim * img_dim  # size of state vector
epsilon = 0.1 # set constant for e_greedy policy 
hidden_size = 256  # size of hidden layer as per assignment spec
flat_size = 7*7*32# convolutional layer output shape hidden layer size
#buff_size = 100000   # size of the replay buffer 
learn_rate = 0.001 # learning rate
start_ep = 100  # number of start episodes to be used

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
file = "PartB2_game:"+env_name
file_save = folder + file
file_N = file + '.npz' # for numpy save/load variables

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

    
##################################--MAIN--###################################

# initialise varaibles
eval_loss = []
eval_len = []
eval_r = []

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

## add bias terms for conv layers:
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
hidd1 = tf.matmul(hidd1_flat, w['Layer1'])   
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
Q_now = tf.matmul(hidd1_now, w['Layer2'])  
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
optimum = tf.train.RMSPropOptimizer(learning_rate = learn_rate)
update = optimum.minimize(loss)

#################################--TRAIN--###################################

with tf.Session() as sess:
    if Train == True:
        sess.run(tf.global_variables_initializer())
        #ReBuff = MakeBuffer2(start_ep) #  start by partialy adding start_episodes to the buffer
        print("\n***   Starting training   *** \n")
        #j = 0 # init global steps counter 
        rewOut = []
        ep_len = []
        for i in range(start_ep) :
            done = False
            loss_ep = 0
            cc = 0  # init local counter
            sum_r = 0.0 # sum of episode rewards
            buff1 = np.ones([state_size,4]) #  init. buffer to store 4 images at a time
            buff2 = buff1    # init. future state buffer1        
            obs = env.reset() ## reset environ.
            obs = Grayscale(obs, img_dim, img_dim) # transform observation to desired shape and grayscale               
            
            for _ in range(4):
                buff1 = Update1(obs, buff1) # update buffer with observation
            
            action = env.action_space.sample() # init. action
            while not done:
                cc += 1
                obs, rew, done, _ = env.step(action)
                obs = Grayscale(obs, img_dim, img_dim)  
                rew = np.clip(rew, -1, 1)
                sum_r = (df ** cc) + rew
                buff2 = Update1(obs, buff1) # update obervations in buffer2
#                if done:
#                    ReBuff.Plus([buff1, action, rew, buff2, 0.0])
#                    break 
#                ReBuff.Plus([buff1, action, rew, buff2, (not done)*1.])
                buff1 = buff2
                if done:
                    break
                # while not done sample another action 
                action = sess.run(max_act, feed_dict={state_now: [buff1]})
                action = MYgreed(0.0, action[0])
            rewOut.append(cc) 
            ep_len.append(sum_r) #mistake wrong result saved to wrong variable, fixed at loading stage
            
            if i%10==0: # sometimes display training results 
                print("episode: ",i, "length: ",cc, "reward: ",sum_r)
        
### Save Trained Model
        # if no path than make path
        if not os.path.exists(folder):
                print('Creating path where to save model: ' + folder)
                os.mkdir(folder)
        saver = tf.train.Saver()    
        saver.save(sess, file_save)  
        print("\nModel saved at: " + file_save)
        # Save results    
        np.savez(file_N, rewOut = rewOut , ep_len = ep_len)
        #
###############################----RESULTS---##################################
#  if not training load saved results:
    else: 
            loaded = np.load(file_N)
            print("Model Loaded\n") 
            print("Paty B 2, game: ",env_name)
            # FIX misate with variable swap by swaping back around:
            ep_len = loaded['rewOut'] # assign episode length
            rewOut = loaded['ep_len'] # assign discounted reward
            
# calc mean and dtandard deviation of score and frames:
ave_score = np.mean(rewOut)
sd_score = np.std(rewOut)
ave_frames = np.mean(ep_len)
sd_frames = np.std(ep_len)
# display results:
print("The Mean Score (i.e discounted reward) is:",ave_score," and the standard deviation is:",sd_score)
print("The Mean Frames (i.e episode length) are:",ave_frames," and their standard deviation is:",sd_frames)


