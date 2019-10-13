# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 18:23:39 2017

@author: klaudia ludwisiak

ADVANCED ML ICA 2
Task 2: Pixel Prediction(many-to-many recurrent model)

The input: current binarised pixel from MNIST digit dataset

The output: output function is now a probability over the value of pixel xt+1 
– which can be either 0 or 1 (black or white)

Different recurrent NN architectures:
(a) GRU with 32, 64, 128 units (scrpit run 3 times for the 3 sizes)
(b) stacked GRU: 3 recurrent layers with 32 units each
"""
# ============================================================================
# ============================ IMPORTANT !!!! ==================================
# SETUP to save/load
Train = False # use this setting to load the model
# if Train = True model will train and save results 
# if Train = False model will LOAD and summary results will be displayed
# BELOW SECTION CALLED CONTROL selects unit size of interest
# ============================================================================
# CONTROL 
Part = 'A' # change to B if want code to run for part B (3xGRU)
# chose unit size (only matters for part A) 
rnn_size = 32 # change to 32/64/128 to run model with different no. units

# ============================================================================
# ============================ End of Choice section =======================

# adjust hyperparameters depending on choice above
if rnn_size == 128:
    batch_size = 500
    learning_rate = 0.0005
if rnn_size == 64:
    batch_size = 400
    learning_rate = 0.004 
if rnn_size == 32:
    batch_size = 250
    learning_rate = 0.02 

if Part == 'B':
    n_layers = 3
    rnn_size = 32
    batch_size = 400
    learning_rate = 0.004
# ============================================================================

# Import required packages 
import matplotlib.pyplot as mtp
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import pickle
import os.path
import numpy as np
np.random.seed(1) # set time seed for same random sample generation

#------------------------------------------------------------------------------
# FUNCTIONS
#------------------------------------------------------------------------------
def Binarise(x, threshold = 0.1):
    return (threshold < x).astype('float32')
# does not improve part 1A results 
    
# ----------------------------------------------------------------------------
# Display loss  vs.training time
def make_plot(inpt, string1):
# string1 is the title reference
    mtp.figure(1)    
    mtp.plot(inpt)
    mtp.title('MNIST - Part 2 GRU '+string1)    
    mtp.ylabel('C.E.Loss per epoch (Training)')
    mtp.xlabel('Num of Epochs')
    mtp.show()
# Display accuracy  vs.training time
def make_plot2(inpt, string1):
# string1 is the title reference
    mtp.figure(1)    
    mtp.plot(inpt)
    
    mtp.title('MNIST - Part 2 GRU '+string1)    
    mtp.ylabel('Training Accuracy')
    mtp.xlabel('Num of Epochs')
    mtp.show()

#------------------------------------------------------------------------------   
#   BUILD GRAPH:
def recurrent_neural_network3(x, drop_prob): 
    # x and drop _prob are placeholders
    
    lin_layer1={'weights1':tf.Variable(tf.truncated_normal([rnn_size, 1],stddev=0.1)),
                  'bias1':tf.Variable(tf.truncated_normal([1]))}
                  
    # initialise cell
    cell = rnn_cell.GRUCell(rnn_size)
    
    # add dropout 
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob = 1, output_keep_prob = drop_prob, seed=1)
    
    if Part == 'B': # for part be add 3 GRU units instead of one
        cell = rnn_cell.MultiRNNCell([cell] * n_layers, state_is_tuple=True)
    
    out, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    cut_out = tf.reshape(out, [tf.shape(out)[0] * (n_strip-1), rnn_size]) 
    # now interested in all but last output of RN
    # (ie. no prediction for last pixel 783)
    
    # linear layer 1
    lin_1 = tf.matmul(tf.reshape(cut_out,[-1, rnn_size]),lin_layer1['weights1']) + lin_layer1 ['bias1']
    
    # batch normalisation    
    bn_w = tf.get_variable('bn_weights', shape=[strip_size, strip_size], initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=1, dtype=tf.float32))
    bn_out = tf.matmul(lin_1, bn_w)

    # batch normalisation 
    bn_1 = batch_normaliz(bn_out)
            
    return bn_1 

#------------------------------------------------------------------------------
# batch normalisation
# adapted from:
# http://r2rt.com/implementing-batch-normalization-in-tensorflow.html

def batch_normaliz(inpt, decay = 0.999, const = 1e-3):
# constant is a small float number to avoid dividing by 0. 
# decay is a parameter which has been kept at default 
    scale = tf.Variable(tf.ones([inpt.get_shape()[-1]]))
    shape = tf.Variable(tf.zeros([inpt.get_shape()[-1]]))
    population_mean = tf.Variable(tf.zeros([inpt.get_shape()[-1]]), trainable=False)
    population_var = tf.Variable(tf.ones([inpt.get_shape()[-1]]), trainable=False)

    batch_mean, batch_var = tf.nn.moments(inpt,[0])
    train_mean = tf.assign(population_mean,
                           population_mean * decay + batch_mean * (1 - decay))
    train_var = tf.assign(population_var,
                          population_var * decay + batch_var * (1 - decay))
    with tf.control_dependencies([train_mean, train_var]):
        return tf.nn.batch_normalization(inpt, batch_mean, batch_var, shape, scale, const)

#------------------------------------------------------------------------------
# SETUP 
#------------------------------------------------------------------------------
# reset graph 
tf.reset_default_graph() # so code can be run multiple times in console and load/save

# Start interactive session
sess = tf.InteractiveSession()

# INITIALISE VARIABLES
strip_size = 1 # sqrt(dim_x) = 28 originally was feeding pixels in lots of 28 to speed up training
n_strip = 784 # now will feed pixel by pixel as per assignment spec
n_epochs = 10
dropout = 0.9 # small dropout helps 

# for loading/ saving:
fileName = "BN_Model_Part2A_GRU_" + str(rnn_size) +'.'+ str(batch_size) +'.'+ str(learning_rate)
if Part == 'B':
    fileName = "BN_Model_Part2B_3GRU_" + str(rnn_size) +'.'+ str(batch_size) +'.'+ str(learning_rate)
folder = "./trained models/"
Model_fileName = folder + fileName 
pickleName= fileName + ".pickle"
P_fileName = folder + pickleName

# load data
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# INITIALISE more VARIABLES
# now y labels are categorical 0/1 and derived from X
train_x = Binarise(mnist.train.images[:,:-1])
train_y = Binarise(mnist.train.images[:,1:])

test_x = Binarise(mnist.test.images[:,:-1])
test_y = Binarise( mnist.test.images[:,1:])

# reshape
train_x = train_x[:, :, np.newaxis]
test_x = test_x[:, :, np.newaxis]

(n_train, dim_x) = (mnist.train.images).shape  
# Num of training data instances is 55000 and dim_x is 784 (pixels per image)
(n_test, dim_y)  = (mnist.test.labels).shape 
# Num of test data instances is 10000 and dim_y is 10
    
# save cross entropy loss/ accuracy over each epoch into this list for plot
trainLossOUT = []
trainAccOUT= []
CELoss =[] #cross entropy loss

#------------------------------------------------------------------------------
# INITIALISE GRAPH  
#------------------------------------------------------------------------------

# INITIALISE TF VARIABLES  
x = tf.placeholder('float32', [None, n_strip-1, strip_size], name='images')
y = tf.placeholder('float32', [None, None], name ='labels')
# add placeholder for dropout
drop_prob = tf.placeholder(tf.float32)

#  MODEL SETUP    
bn = recurrent_neural_network3(x,drop_prob)
# use RNN output to calc cross entropy loss (cost fun):
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(bn, y)) 
# define optimiser and set objectuve to minimise cost
adam = tf.train.AdamOptimizer(learning_rate=learning_rate) 
optimum = adam.minimize(cost)
# find y predictions and accuracy
y_prediction = tf.cast(tf.nn.sigmoid(bn)>0.5,tf.float32) # make into 1 or 0
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_prediction, y), 'float32'))

#------------------------------------------------------------------------------
#  TRAIN MODEL
#------------------------------------------------------------------------------
# INITIALISE SAVER
saver = tf.train.Saver()

if Train == True:
 
    print('Commence training '+fileName+' epochs: '+str(n_epochs))
    # run session
    sess.run(tf.global_variables_initializer())
        
    for epoch in range(n_epochs):

        for run in range(int(mnist.train.num_examples/batch_size)):
            # select x and y batches
            batch_x = train_x[run * batch_size: (run + 1) * batch_size]
            batch_y = train_y[run * batch_size: (run + 1) * batch_size]
            # reshape
            #batch_x = batch_x.reshape(batch_size,(n_strip-1), 1)          
            batch_y = batch_y.reshape(batch_size*(n_strip-1), 1)
            # run optimisation
            sess.run(optimum, feed_dict = {x: batch_x, y: batch_y, drop_prob: dropout})
        
        # get results each epoch    
        cst, acc = sess.run([cost, accuracy], feed_dict = {x: batch_x, y: batch_y, drop_prob :1}) 
        
        CELoss.append(cst) # store cross entr. loss after each epoch
        trainAccOUT.append(acc)  

        print('Training Epoch', epoch, 'C.E.Loss:',cst, 'accuracy:',acc)
  
  # TEST
    test_y = np.reshape(test_y, [n_test * (n_strip-1), 1])
               
    # check if prediction was correct and find accuracy
    testloss = cost.eval({x: test_x, y: test_y, drop_prob :1})
    testacc = accuracy.eval({x: test_x, y: test_y, drop_prob :1})
    print('Test loss:',testloss,'Accuracy:', testacc) 

    if Part == 'A':
    # plot data
        make_plot(CELoss,str(rnn_size))
        make_plot2(trainAccOUT,str(rnn_size))

    if Part == 'B':
    # plot data
        make_plot(CELoss,str(rnn_size)+' with 3 layers')
        make_plot2(trainAccOUT,str(rnn_size)+' with 3 layers')

## ---------------------------- LOAD MODEL -----------------------------------   
else:  
    print('Commence loading:'+Model_fileName)
    saver.restore(sess,Model_fileName)
    print('\nModel successfully loaded.\n')        
    print('Calculating train/test loss and accuracy')
    
    train_y = np.reshape(train_y, [n_train * (n_strip-1), 1])
    
    trainloss1 = cost.eval({x: train_x, y: train_y, drop_prob :1.0})
    print(trainloss1)
    trainacc1 = accuracy.eval({x: train_x, y: train_y, drop_prob :1.0})
    
    test_y = np.reshape(test_y, [n_test * (n_strip-1), 1])

    testloss1 = cost.eval({x: test_x, y: test_y, drop_prob :1.0})
    testacc1 = accuracy.eval({x: test_x, y: test_y, drop_prob :1.0})

#    trainloss1 = cost.eval({x: train_x.reshape((-1, n_strip, strip_size)), y: train_y})
#    trainacc1 = accuracy.eval({x: train_x.reshape((-1, n_strip, strip_size)), y: train_y})
#    testloss1 = cost.eval({x: test_x.reshape((-1, n_strip, strip_size)), y: test_y})
#    testacc1 = accuracy.eval({x: test_x.reshape((-1, n_strip, strip_size)), y: test_y})
    print('\nAfter ',n_epochs,' epochs the final training C.E.Loss:', trainloss1,'and Accuracy:', trainacc1) 
    print('\nThe corresponding test Loss:',testloss1,'and Accuracy:', testacc1) 
    
    with open(P_fileName, 'rb') as f:
        print('\nEnhanced results summary from pickle file:')
        testloss, testacc, trainAccOUT, CELoss = pickle.load(f)
        
        if Part == 'A':
        # plot data
            make_plot(CELoss,str(rnn_size))
            make_plot2(trainAccOUT,str(rnn_size))

        if Part == 'B':
        # plot data
            make_plot(CELoss,str(rnn_size)+' with 3 layers')
            make_plot2(trainAccOUT,str(rnn_size)+' with 3 layers')
        
## ---------------------------- SAVE MODEL -----------------------------------   
  
# if no path than make path
if not os.path.exists(folder):
        print('Creating path where to save model: ' + folder)
        os.mkdir(folder)

# save model and selected variables
print('Saving model at: ' + fileName)
saver.save(sess, Model_fileName)
print('Model successfully saved.\n')        
#Save_Model(sess,  MODEL_FILENAME)
with open(P_fileName, 'wb') as f:
   pickle.dump([testloss, testacc, trainAccOUT, CELoss], f, protocol=-1)

#close session
sess.close()  

