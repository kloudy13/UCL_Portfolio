# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 19:00:23 2017


@author: klaudia ludwisiak

ADVANCED ML CW1 PART 1 

For the task specification refer to Assignment1_Task.pdf

 code description 
 this script solves part 1 of assignment and was built around the model(s)
 architectures which will be implemented. It was noted that part A constitutes 
 of a linear layer followed by a softmax, which was implemented as a function. 
 Parts B and C expand on this by introducing a nonlinear ReLU layer before the 
 linear layer (from part A), these was added using anther helper function.
 Finally, Part D a 2 layer CNN with maxpool was implemented, this was flattened  
 and also passed through ReLu and linear layer similar to parts B/C. 
"""

# import packages
import tensorflow as tf
import matplotlib.pyplot as mtp
import math
import numpy as np
import os.path

# ========================== IMPORTANT !!!! ==================================
# SETUP to save/load
Train = False
Load = True # use this setting to load the model
# if Train = True model will train and save results 
# ============================================================================

# ====================== Functions ===========================================

#-Build Part A model---------------------------------------------------------------------------
def Part_A_Model(x, dim1, dim2, sd):
# dim1 is dimenssion in and dim2 is dimenssion out    

   # output layer
    out = LinLayer(x, dim1, dim2, sd, bias = True)
    return out 
    
#-Build Part B model---------------------------------------------------------------------------

def Part_B_Model(x, dim1, dim2, dimH, sd):
# Imputs:
# dim1 is dimenssion in and dim2 is dimenssion out 
# dimH is internal hidden layer param.
               
        # hidden layer 
        h = HiddLayer(x, dim1, dimH, sd)#, dropout = True, dropout_prob=0)
        
        # linear layer& output 
        out = LinLayer(h, dimH, dim2, sd, bias = True)
       
        return out
        
#-Build Part C model---------------------------------------------------------------------------
def Part_C_Model(x, dim1, dim2,  dimH1, dimH2, sd):
# Imputs:
# dim1 is dimenssion in (x) and dim2 (y) is dimenssion out 
# dimH1, dimH2 are internal hidden layer param.

        # hidden layer1
        h1 = HiddLayer(x, dim1, dimH1, sd)#, dropout=True, dropout_prob=0)#0.5)
       
        # hidden layer2
        h2 = HiddLayer(h1, dimH1, dimH2, sd =0.01)#, dropout=True, dropout_prob=0)#0.7)
        
        # linear layer & output
        out = LinLayer(h2, dimH2, dim2, sd =0.01, bias=True)

        return out

#-Build Part D model------------------------------------------------------------
        
def Part_D_Model(x, dim1, dim2, dimHD, shape_conv1, shape_conv2 , stridesConv, stridesPool, sd, chanel):
# Returns model: Conv1 -> maxpool1-> conv2 -> maxpool2 -> flatten -> ReLU -> Linear layer
# Imputs:
# x is data tensor, channel is the image channel (1 == grayscale), 
# sd id the standard deviation to be used 
# dim1, dim 2 are the numbder of images and lables in the data respectievely
# dimH is internal hidden ReLU layer parameter
# shape_conv1, shape_conv2 are 4D dimensions of convolution
# stridesConv, stridesPool are stride parameters (if not equal 1 then skip information)
    
    # reshape
    dim_image = int(math.sqrt(dim1))
    x = tf.reshape(x, [-1, dim_image, dim_image, chanel]) 
    
    # Convolution layer 1
    w1 = tf.Variable(tf.random_normal(shape_conv1, stddev = sd))
    conv_layer1 = tf.nn.relu(tf.nn.conv2d(x ,w1, strides = stridesConv, padding='SAME'))
   
    # maxpool 1    
    out1 = tf.nn.max_pool(conv_layer1, ksize=[1,2,2,1], strides = stridesPool, padding='SAME')
    
    # Convolution layer 2
    w2 = tf.Variable(tf.random_normal(shape_conv2, stddev = sd))
    conv_layer2 = tf.nn.relu(tf.nn.conv2d(out1, w2, strides = stridesConv, padding='SAME'))
  
    # maxpool 2     
    out2 = tf.nn.max_pool(conv_layer2, ksize=[1,2,2,1], strides = stridesPool, padding='SAME')
    
    # Flatten output to change shape to 1D vector 4*4*128  
    h_flat = tf.reshape(out2, [-1,64*7*7]) 
	# max pool reduced sdiemsnion by a factor of two each time this si why now need 7 by 7

    # Fully-connected hidden layer (ReLU)  
    h = HiddLayer(h_flat, 64*7*7, dimHD, sd)
    
    # linear Layer and final output    
    out = LinLayer(h, dimHD, dim2, sd)
    
    return out
    
# ----------------------------------------------------------------------------
#  ReLU hidden layer
def HiddLayer(x, dim1, dim2, sd):
	# dim1 is dimenssion in(x dim) and dim2(y dim) is dimenssion out
      # sd is the standard deviation

	# ReLU hidden layer	
    w_h = genRandVar(dim1, dim2, sd)
    b_h = genRandVar(1, dim2, sd)
    h = tf.nn.relu(tf.matmul(x, w_h)+b_h)
    return h

# -----------------------------------------------------------------------------
# Linear layer + w/ or w/o bias compoment
def LinLayer(x, dim1, dim2, sd, bias=True):
    # dim1 is dimenssion in and dim2 is dimenssion out
    # sd is the standard deviation

	if bias:
         b_l = tf.constant(sd,shape=[1,dim2])
         w_l = genRandVar(dim1, dim2, sd)

         return tf.matmul(x, w_l)+b_l
	else:
		w_l = genRandVar(dim1, dim2, sd)
		return tf.matmul(x, w_l)

#------------------------------------------------------------------------------
# Generate tf random varaibales of required dimensions
def genRandVar( dim1, dim2, sd):
     # dim1 is dimenssion in and dim2 is dimenssion out
     # sd is the standard deviation
     # var = tf.Variable(tf.random_normal([dim1, dim2], sd))
      var = tf.Variable(tf.truncated_normal([dim1, dim2],stddev=sd)/np.sqrt([dim1, dim2]).sum())
      return var
  
#------------------------------------------------------------------------------
# function to run the model and close the session
# output is a list of accuracy and loss for test and train sets
def RunModel3(sess, acc, loss, max_iter, dim_batch, train_out,  x, y, train_x, train_y, test_x, test_y, folder, fileName, string):
  
  accTrain, accTest = [],[]	
  lossTrain, lossTest = [],[]
  for i in range(max_iter): # loop over batches 
               
                for startIndex, endIndex in zip(range(0,len(train_x),dim_batch), range(dim_batch,len(train_x),dim_batch)):
                     sess.run(train_out, feed_dict = {x: train_x[startIndex:endIndex], y: train_y[startIndex:endIndex]})
                     
                if i%1==0 : # save var. for plotting every iteration
                     acc_train = sess.run(acc, feed_dict = {x:train_x, y:train_y})
                     acc_test  = sess.run(acc, feed_dict = {x:test_x,  y:test_y})
                     loss_train = sess.run(loss, feed_dict = {x:train_x, y:train_y})
                     loss_test = sess.run(loss, feed_dict = {x:test_x, y:test_y})
                     
                     accTrain.append(acc_train)
                     accTest.append(acc_test)
                     lossTrain.append(loss_train)
                     lossTest.append(loss_test)
 
#			# Print accuracy and loss every 10 iterations
                if i%10==0 :
                     print('Iteration %d: Accuracy %.3f(train) %.3f(test)' %(i, acc_train, acc_test))
                     print('Iteration %d: loss %.3f(train) loss %.3f(test)' %(i,  loss_train, loss_test))
                  
  Save_Model(sess, folder, fileName, string)                    
                    
  return(accTrain, accTest, lossTrain, lossTest)

#------------------------------------------------------------------------------
# Display accuracy and loss vs.training time
def make_plot(acc, loss, string1, string2):
# string(1,2) is the title reference
    mtp.figure(1)    
    mtp.subplot(211)
    mtp.plot(acc)
    mtp.title('MNIST - Part '+string1+string2)    
    mtp.ylabel('Accuracy')
    mtp.subplot(212)
    mtp.plot(loss)
    mtp.ylabel('Cross Entropy Loss')
    mtp.xlabel('Num of Epochs')
    mtp.show()

#------------------------------------------------------------------------------
# make confussion matrix - needs list of np.arrays as imput (labels var)
def conf_mat(y_pred, y_true):
    # initialise the confusion matrix
    confusion = np.zeros([10,10])
    for c in range(0, y_true.shape[0]):
        # adjust format of y_pred to categorical (select max value and =1 all else =0)
        idx = np.unravel_index(y_pred[c].argmax(),y_pred[c].shape)
        y_pred[c,idx] = 1
        for i in range( len(y_pred[c])):
            if y_pred[c,i] != 1:
                y_pred[c,i] =0
            
        # check if all elemnets in array row match (if yes then correct)
        if y_true[c].all() == y_pred[c].any():
            idx = np.unravel_index(y_pred[c].argmax(), y_pred[c].shape)
            confusion[idx, idx] += 1
        else:
            idx_pred = np.unravel_index(y_pred[c].argmax(), y_pred[c].shape)
            idx_true = np.unravel_index(y_true[c].argmax(), y_true[c].shape)
            confusion[idx_true, idx_pred] += 1

    print('\n Confusion Matrix: \n')
    print(confusion.astype(int))
    print('\n Horizontal axis: Predicted Labels\n Vertical axis: True Labels')
    return confusion.astype(int)
    

#------------------------------------------------------------------------------
#Function to save model

def Save_Model(sess, folder, fileName, string):
         if not os.path.exists(folder):
             print( 'Creating path to save model: ' + folder)
             os.mkdir(folder)

         print('Saving model at: '+ string + fileName)
         saver = tf.train.Saver()
         saver.save(sess, string + fileName)
         print('Model succesfully saved.\n')

#------------------------------------------------------------------------------
#Function to load model
def Load_Model(sess, folder, fileName):
    if os.path.exists(folder):
        print('Loading saved model')# from: ' folder)
        saver = tf.train.Saver()
        saver.restore(sess, fileName)
        print('Model succesfully loaded.\n')       
        return True
    elif os.path.exists(fileName):
        print('Loading saved model')# from: ' folder)
        saver = tf.train.Saver()
        saver.restore(sess, fileName)
        print('Model succesfully loaded.\n')
        return True
    else:
        print('Model file does not exists!')
        return False

def Load_Model2(sess, fileName):

    if os.path.exists(fileName):
        print('Loading saved model')# from: ' folder)
        saver = tf.train.Saver()
        saver.restore(sess, fileName)
        print('Model succesfully loaded.\n')
        return True
    else:
        print('Model file does not exists!')
        return False

# ========================= PART == 1 === SOLUTION ============================

# ============================ Load dataset ===================================

from tensorflow.examples.tutorials.mnist import input_data
# Import dataset: 
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train_x = mnist.train.images # images
train_y =  mnist.train.labels # labels
test_x = mnist.test.images
test_y = mnist.test.labels

# could include a validation set but not essential 

(Train_dat_inst, dim_x) = train_x.shape  
# Num of trainig data instances is 55000 and dim_x is 784
(Test_dat_inst, dim_y)  = test_y.shape 
# Num of test data instances is 10000 and dim_y is 10

# ======================== MAKE MODEL--> build graph ==========================
# Initialise placeholders and variables
x = tf.placeholder(tf.float32, [None, dim_x])
y = tf.placeholder(tf.float32, [None, dim_y])

# ------------------------------ TRAIN MODELS ---------------------------------
### PART __A 
# 1 linear layer, followed by a softmax.

sdA = 0.1
y_model_A = Part_A_Model(x, dim_x, dim_y, sdA)

### PART __B
# 1 hidden layer (128 units) with a ReLU non-linearity, followed by a softmax.

# additional parametres:
dimH = 128  # specify size of ReLU
sdB = 2/math.sqrt(dimH) # standard dev. rule of thumb

# model:
y_model_B = Part_B_Model(x, dim_x, dim_y, dimH, sdB)

### PART __C
# 2 hidden layers (256 units), with ReLU non-linearity, follow by a softmax.

# additional parametres:
dimH1, dimH2 = 256, 256  # specify size of ReLU
sdC = 2/math.sqrt(dimH1)

# model:
y_model_C = Part_C_Model(x, dim_x, dim_y, dimH1, dimH2, sdC)

### PART __D
# 3 layer convolutional model (2 convolutional layers followed by max pooling) 
# + 1 non-linear layer (256 units), followed by softmax.

# additional parametres:
dimHd = 256  # specify size of ReLU
chanel = 1 # num of image chanels (1 is grayscale)
sdD = 2/math.sqrt(dimHd) # standard dev. rule of thumb
shape_conv1 = [3,3,1,32] # shape of conv layer
shape_conv2 = [3,3,32,64]
stridesConv= [1,1,1,1] # stride 
stridesPool= [1,2,2,1]

# model:
#y_model_D = Part_D_Model(x, chanel, dim_x, dim_y,  dimHd, sdD)
y_model_D = Part_D_Model(x, dim_x, dim_y, dimHd, shape_conv1, shape_conv2 , stridesConv, stridesPool, sdD, chanel)

# ---------------- Assert correctness of predicted lables ---------------------

y_corr_A = tf.equal(tf.argmax(y_model_A, 1), tf.argmax(y,1))
y_corr_B = tf.equal(tf.argmax(y_model_B, 1), tf.argmax(y,1))
y_corr_C = tf.equal(tf.argmax(y_model_C, 1), tf.argmax(y,1))
y_corr_D = tf.equal(tf.argmax(y_model_D, 1), tf.argmax(y,1))

# --------------------- Cross-Entropy Function --------------------------------
# calculate softmax at same time for numerical stabilty

cross_ent_loss_A = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_model_A,logits= y))
cross_ent_loss_B = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_model_B,logits= y))
cross_ent_loss_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_model_C,logits= y))
cross_ent_loss_D = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_model_D,logits= y))

# -------------------------- Measure accuracy --------------------------------- 
# comare predicted lables with real ones:

acc_A = tf.reduce_mean(tf.cast(y_corr_A, tf.float32 ))
acc_B = tf.reduce_mean(tf.cast(y_corr_B, tf.float32 ))
acc_C = tf.reduce_mean(tf.cast(y_corr_C, tf.float32 ))
acc_D = tf.reduce_mean(tf.cast(y_corr_D, tf.float32 ))

# ======================== TRAIN MODEL(s)======================================
# Additional SETUP
# parameters :
rateA, rateB, rateC, rateD = 0.3, 0.2, 0.2, 0.2 # affects convergence
dim_batch = 50 # data size is divisible by 50
# set time seed: 
tf.set_random_seed(1)
# specify save location and filename (for each model separte save)
folder   = "/Users/klaudia/Documents/UCL/Advanced ML/CW1"
fileName = "Part1.ckpt" 

# Optimise 
train_out_A = tf.train.GradientDescentOptimizer(rateA).minimize(cross_ent_loss_A) 
train_out_B = tf.train.GradientDescentOptimizer(rateB).minimize(cross_ent_loss_B) 
train_out_C = tf.train.GradientDescentOptimizer(rateC).minimize(cross_ent_loss_C) 
train_out_D = tf.train.GradientDescentOptimizer(rateD).minimize(cross_ent_loss_D) 

# -------------------------- Run Models(s) ------------------------------------
# Start TF session 
with tf.Session() as sess:
# Initialise variables
    tf.global_variables_initializer().run() 

if Train == True:    
        
        # PART A
        # set epochs:
        max_iter = 20
        string="A_" # need for file saving 
        RAaccTrain, RAaccTest, RAlossTrain, RAlossTest = RunModel3(sess, acc_A, cross_ent_loss_A, max_iter, dim_batch, train_out_A, x, y, train_x, train_y, test_x, test_y, folder, fileName, string)
    
        # PART B
        # set epochs:
        max_iter = 20
        string = "B_"    
        RBaccTrain, RBaccTest, RBlossTrain, RBlossTest = RunModel3(sess, acc_B, cross_ent_loss_B, max_iter, dim_batch, train_out_B, x, y, train_x, train_y, test_x, test_y,folder, fileName, string)
    
        # PART C
        # set epochs:
        max_iter = 15
        string = "C_"
        RCaccTrain, RCaccTest, RClossTrain, RClossTest = RunModel3(sess, acc_C, cross_ent_loss_C, max_iter, dim_batch, train_out_C, x, y, train_x, train_y, test_x, test_y,folder, fileName, string)
    
        # PART D
        # reduce epochs:
        max_iter = 8
        string = "D_"
        RDaccTrain, RDaccTest, RDlossTrain, RDlossTest = RunModel3(sess, acc_D, cross_ent_loss_D, max_iter, dim_batch, train_out_D, x, y, train_x, train_y, test_x, test_y,folder, fileName, string)
    

# ============================ IMPORTANT!!! ===================================
if Load == True:
# set folder to local path
## ============================ Load model ====================================
   with tf.Session() as sess:
       tf.global_variables_initializer().run()
       
       foleder = "this should be set to user path"
       fileNameA = "A_Part1.ckpt.meta"
       fileNameB = "B_Part1.ckpt.meta"
       fileNameC = "C_Part1.ckpt.meta"
       fileNameD = "D_Part1.ckpt.meta"
       # 
       Load_Model(sess,folder ,fileNameA)
       Load_Model(sess,folder ,fileNameB)
       Load_Model(sess,folder ,fileNameC)
       Load_Model(sess,folder ,fileNameD)

### ISSUES WITH LOADING 
       # can also try the modified load function below:
       #Load_Model2(sess ,fileNameA)
# Generally, due to the way in which my RunModel function works it looks to be 
# impossible to load the models. I woudl be appreciative if you could still
# run my code (setting Train = True and Load =False at the top of the script).
# my model runs runs in minutes. 

## ---------------------------- PLOT RESULTS ----------------------------------
#
mtp.figure(1)    
mtp.subplot(4,1,1)
make_plot(RAaccTrain, RAlossTrain, string1 = '1.A', string2 = ' Train')
mtp.subplot(4,1,2)
make_plot(RAaccTest, RAlossTest, string1 = '1.A', string2 =' Test')
mtp.subplot(4,2,1)
make_plot(RBaccTrain, RBlossTrain, string1 = '1.B',string2 = ' Train')
mtp.subplot(4,2,2)
make_plot(RBaccTest, RBlossTest, string1 = '1.B', string2 =' Test')
mtp.subplot(4,3,1)
make_plot(RCaccTrain, RClossTrain, string1 = '1.C',string2 = ' Train')
mtp.subplot(4,3,2)
make_plot(RCaccTest, RClossTest, string1 = '1.C', string2 =' Test')
mtp.subplot(4,4,1)
make_plot(RDaccTrain, RDlossTrain, string1 = '1.D',string2 = ' Train')
mtp.subplot(4,4,2)
make_plot(RDaccTest, RDlossTest, string1 = '1.D', string2 =' Test')



