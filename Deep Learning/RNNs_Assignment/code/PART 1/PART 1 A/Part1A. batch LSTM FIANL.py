# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 18:23:39 2017

@author: klaudia ludwisiak

ADVANCED ML ICA 2

using TensorFlow v 0.12

Task 1: Classification

The input: current binarised pixel from MNIST digit dataset

The output: probabilities over the 10 classes 
(RNN -> affine transformation (linear layer - 100 neurons) + non-linearity...
(ReLU) + affine transformation + softmax layer.)

Different recurrent NN architectures:
(a) LSTM with 32, 64, 128 units (scrpit run 3 times for the 3 sizes)
(b) GRU with 32, 64, 128 units (scrpit run 3 times for the 3 sizes)
(c) stacked LSTM: 3 recurrent layers with 32 units each 
(d) stacked GRU: 3 recurrent layers with 32 units each

code for each part provided separately

"""

# ============================================================================
# ========================== IMPORTANT !!!! ==================================
# SETUP to save/load
Train =  False# use this setting to load the model
# if Train = True model will train and save results 
# if Train = False model will LOAD and summary results will be displayed
# BELOW SECTION CALLED CONTROL selects unit size of interest
# ============================================================================
# CONTROL 
# chose units 
rnn_size = 64 # change to 32/64/128 to run model with different no. units
# adjust hyperparameters
if rnn_size == 128:
    batch_size = 500
    learning_rate = 0.0005 
if rnn_size == 64:
    batch_size = 400
    learning_rate = 0.004 
if rnn_size == 32:
    batch_size = 250
    learning_rate = 0.018 
# ============================================================================
# ============================================================================

# Import required packages:
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
    mtp.title('MNIST - Part 1A LSTM '+string1)    
    mtp.ylabel('C.E.Loss per epoch (Training)')
    mtp.xlabel('Num of Epochs')
    mtp.show()
# Display accuracy  vs.training time
def make_plot2(inpt, string1):
# string1 is the title reference
    mtp.figure(1)    
    mtp.plot(inpt)
    mtp.title('MNIST - Part 1A LSTM '+string1)    
    mtp.ylabel('Training Accuracy')
    mtp.xlabel('Num of Epochs')
    mtp.show()

#------------------------------------------------------------------------------   
#   BUILD GRAPH:
def recurrent_neural_network3(x,y): 
    # x and y are placeholders
    
    #initialise params for linear layers     
    lin_layer1={'weights1':tf.Variable(tf.truncated_normal([rnn_size, 100],stddev=0.1)),
                  'bias1':tf.Variable(tf.truncated_normal([100]))}

    lin_layer2={'weights':tf.Variable(tf.truncated_normal([100, dim_y],stddev=0.1)),
                    'bias':tf.Variable(tf.truncated_normal([dim_y]))}
    
    # reshape input data
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, strip_size])
    x = tf.split(0, n_strip, x)

    # initialise cell
    cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
    
    out, _ = rnn.rnn(cell, x, dtype=tf.float32)
    
    last_out = out[-1] # only interested in last output of RNN (many to one)
    
    # linear layer 1
    lin_1 = tf.matmul(last_out,lin_layer1['weights1']) + lin_layer1 ['bias1']
    
    # batch normalisation 
    bn_1 = batch_normaliz(lin_1)
    
    # ReLU 
    hid_1 = tf.nn.relu(bn_1)
    
    # linear layer 2
    lin_2 = tf.matmul(hid_1,lin_layer2['weights']) + lin_layer2['bias']
        
    return lin_2 # == y_pred

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

tf.reset_default_graph() # reset graph so code can be run multiple times in console and load/save

# Start interactive session
sess = tf.InteractiveSession()


# INITIALISE VARIABLES
strip_size =  1 # sqrt(dim_x) = 28 originally was feeding pixels in lots of 28 to speed up training
n_strip = 784 # now will feed pixel by pixel as per assignment spec
n_epochs = 18

# for loading/ saving:
fileName = "BN_Model_Part1A_" + str(rnn_size) +'.'+ str(batch_size) +'.'+ str(learning_rate)
folder = "./trained models/"
Model_fileName = folder + fileName 
pickleName= fileName + ".pickle"
P_fileName = folder + pickleName

# load data
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# INITIALISE more VARIABLES
test_y = mnist.test.labels
train_x = Binarise(mnist.train.images)
train_y = mnist.train.labels
test_x = Binarise(mnist.test.images)

(n_train, dim_x) = (mnist.train.images).shape  
# Num of training data instances is 55000 and dim_x is 784 (pixels per image)
(_, dim_y)  = (mnist.test.labels).shape 
# Num of test data instances is 10000 and dim_y is 10
    
# save cross entropy loss/ accuracy over each epoch into this list for plot
trainLossOUT = []
trainAccOUT= []
CELoss =[] #cross entropy loss

#------------------------------------------------------------------------------
# INITIALISE GRAPH  
#------------------------------------------------------------------------------

# INITIALISE TF VARIABLES  
x = tf.placeholder('float32', [None, n_strip, strip_size], name='images')
y = tf.placeholder('float32', name ='labels')

#  MODEL SETUP    
y_prediction = recurrent_neural_network3(x,y)
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = y_prediction, labels = y) ) 
optimum = tf.train.AdamOptimizer(learning_rate=learning_rate) 
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_prediction, 1), tf.argmax(y, 1)), 'float'))

# gradient clipping
temp = optimum.compute_gradients(cost)
clipped_temp = [(tf.clip_by_value(grad, -0.9, 0.9), var) for grad, var in temp] 
    # clipping params chosen after some trial and error  
clipped_optimum = optimum.apply_gradients(clipped_temp)

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
        loss = 0 # epoch loss (sum of corss entropy loss over epoch)
        for _ in range(int(mnist.train.num_examples/batch_size)):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = batch_x.reshape((batch_size, n_strip, strip_size))
            batch_x = Binarise(batch_x) #seems to make results worse
            _, cst = sess.run([ clipped_optimum, cost], feed_dict={x: batch_x, y: batch_y})
            loss += cst # epoch loss
        
        trainLossOUT.append(loss) # store epoch loss in list 
        CELoss.append(cst) # store cross entr. loss after each epoch

        Rtrain_x = train_x.reshape(-1, n_strip, strip_size)
        acc = sess.run(accuracy, feed_dict={x: Rtrain_x, y: train_y})

        trainAccOUT.append(acc)  

        print('Training Epoch', epoch,'epoch loss:',loss,'C.E.Loss:',cst, 'accuracy:',acc)
                    
    # check if prediction was correct and find accuracy
    testacc = accuracy.eval({x: test_x.reshape((-1, n_strip, strip_size)), y: test_y})
    testloss = cost.eval({x: test_x.reshape((-1, n_strip, strip_size)), y: test_y})
        
    print('Test loss:',testloss,'Accuracy:', testacc) 
    
    #plot data
    make_plot(CELoss,str(rnn_size))
    make_plot2(trainAccOUT,str(rnn_size))

## ---------------------------- LOAD MODEL -----------------------------------   
else:  
    print('Commence loading:'+Model_fileName)
    saver.restore(sess,Model_fileName)
    print('\nModel successfully loaded.\n')        
    print('Calculating train/test loss and accuracy')
    trainloss1 = cost.eval({x: train_x.reshape((-1, n_strip, strip_size)), y: train_y})
    trainacc1 = accuracy.eval({x: train_x.reshape((-1, n_strip, strip_size)), y: train_y})
    testloss1 = cost.eval({x: test_x.reshape((-1, n_strip, strip_size)), y: test_y})
    testacc1 = accuracy.eval({x: test_x.reshape((-1, n_strip, strip_size)), y: test_y})
    print('\nAfter ',n_epochs,' epochs the final training C.E.Loss:', trainloss1,'and Accuracy:', trainacc1) 
    print('\nThe corresponding test Loss:',testloss1,'and Accuracy:', testacc1) 
    
    with open(P_fileName, 'rb') as f:
        print('\nEnhanced results summary from pickle file:')
        testloss, testacc, trainLossOUT, trainAccOUT, CELoss = pickle.load(f)
                
        # plot data
        make_plot(CELoss,str(rnn_size))
        make_plot2(trainAccOUT,str(rnn_size))
  
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
   pickle.dump([testloss, testacc, trainLossOUT, trainAccOUT, CELoss], f, protocol=-1)

#close session
sess.close()  