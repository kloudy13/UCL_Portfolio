# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 16:49:33 2017

@author: klaudia

ADVANCED ML CW1 PART 2

"""
# code description  
# In this part of the assignment I implement part 1 without using tensor flow

# import packages
import matplotlib.pyplot as mtp
import numpy as np
import dill

# ========================== IMPORTANT !!!! ==================================
# SETUP to save/load
Train = False # use this setting to load the model
 # if Train = True model will train and save results 
# ============================================================================

# ------------------------- CLASSES -------------------------------------------
###### LINEAR LAYER 

class LinLayer(object):
    def __init__(self,dim1, dim2, sd = 0.01):
    # dim 1/2 is the num of features/labels
    #sd id the standard deviation
        # initialsie weights and biases:
        np.random.seed(1) # set time seed to replicate results
        self.name = "linear layer"        
        self.w = sd * np.random.randn(dim1, dim2) #np.zeros(shape=[dim1, dim2])  # model weights
        self.b = sd * np.random.randn(dim2) #np.zeros(shape=[dim1])  # model bias
        # initialise other:        
        self.x = None # data
        self.y_pred = None # predicted lables
        self.ddx = None 
        self.ddw = None # deriv wrt w
        self.ddb = None # deriv wrt b

    def ForPass(self, x):
        y_pred = np.add(np.dot(x, self.w), self.b)
        self.y_pred = y_pred
        self.x = x
        return (self.y_pred)

    def BackPass(self, ddy):
        self.ddx = np.dot(ddy, self.w.T) #().T is the transpose
        return (self.ddx)

    def Gradient(self, ddy):
        self.ddw = (self.x.T).dot(ddy)
        self.ddb = ddy.sum(axis=0)
        return (self.ddw, self.ddb)

    def Update(self, rate): #ddw, ddb, rate):
        self.w = self.w - (rate / np.shape(self.x)[0]) * self.ddw
        self.b = self.b - (rate / np.shape(self.x)[0]) * self.ddb
        return (self.w, self.b)


###### SOFT MAX CROSS ENTROPY

class SoftMax_CrossEnt(object):
    def __init__(self):
        self.name = "Softmax with Cross Entropy"
        self.cross_ent = None
        self.y_pred = None
        self.y = None

    def ForPass(self, y,  y_pred):
        
        self.y_pred = y_pred
        exp_y_pred = np.exp(y_pred)
        sum_exp_y_pred = np.sum(exp_y_pred, axis=1,keepdims=True)
        y_pred = exp_y_pred/ sum_exp_y_pred
        self.y_pred = y_pred
       
        self.y = y
        cross_ent = - np.sum(np.multiply(self.y, np.log(y_pred)))  
        self.cross_ent = cross_ent

        return (self.y_pred, self.cross_ent)

    def BackPass(self):
        return (self.y_pred - self.y)

# ---------------------------- OTHER FUNCTIONS -------------------------------- 
# calculate accuracy 
def Accur(y,  y_pred):
    # check if true and predicted y are the same (accuracy)
    acc = np.sum(np.equal(np.argmax(y_pred, axis=1), np.argmax(y, axis=1))) / float(np.shape(y)[0])  
    return (acc)

# ----------------------------------------------------------------------------

# Display accuracy vs.training time
def make_plot2(acc, string1, string2):
# string(1,2) is the title reference
    mtp.figure(1)    
    mtp.plot(acc)
    mtp.title('MNIST - Part2'+string1+string2)    
    mtp.ylabel('Accuracy')
    mtp.xlabel('Num of Epochs')
    mtp.show()
    
# ----------------------------------------------------------------------------
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
def Save_Model(fileName):
    #Saves the model  
    dill.dump_session(fileName)
    print('Model succesfully saved.\n')

#------------------------------------------------------------------------------
#Function to load model
def Load_Model(fileName):
    dill.load_session(fileName)
    print('Model succesfully loaded.\n')
    return


# SETUP to save/load
if Train == True:

# ========================= PART == 2B === SOLUTION ============================
# 1 linear layer, followed by a softmax

# ============================ Load dataset ===================================
### Setup

    from tensorflow.examples.tutorials.mnist import input_data
    # Import dataset: 
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    train_x = mnist.train.images
    train_y = mnist.train.labels
    test_x = mnist.test.images
    test_y = mnist.test.labels
    
    # could include a validation set but not essential 
    
    (Train_dat_inst, dim_x) = train_x.shape  
    # Num of trainig data instances is 55000 and dim_x is 784
    (Test_dat_inst, dim_y)  = test_y.shape 
    # Num of test data instances is 10000 and dim_y is 10
    
    # Other setup;
    rate = 0.2 # learning rate
    dim_batch = 500  # batch size
    max_iter = 35 # epochs
    
    # initialise lists to be stored
    trainAcc = []
    testAcc = []
       
### Define the neural network for part 2B: Linear layer +softmax
    
    # imput layer
    LinLayer_out =  LinLayer(dim_x, dim_y) 
    # Output layer
    SoftMax_CrossEnt_out = SoftMax_CrossEnt()

### TRAIN
    
    for i in range(max_iter):
        # SGD so need batching:
        for batch in range(Train_dat_inst//dim_batch): #split data into batches
            batch_train_x = train_x[ batch * dim_batch : (batch+1) * dim_batch]
            batch_train_y = train_y[ batch * dim_batch : (batch+1) * dim_batch]
           
           # pass forward
            activation = LinLayer_out.ForPass(batch_train_x)
            train_y_pred, train_loss = SoftMax_CrossEnt_out.ForPass(batch_train_y ,activation)
        
            # pass backward
            bck = SoftMax_CrossEnt_out.BackPass()
        
            # Gradient descent ==> train model 
            ddW, ddb = LinLayer_out.Gradient(bck)
            w, b = LinLayer_out.Update(rate)
        
        # pass forward
        activation = LinLayer_out.ForPass(train_x)
        train_y_pred, train_loss = SoftMax_CrossEnt_out.ForPass(train_y ,activation)
        
        #test set
        activation2 = LinLayer_out.ForPass(test_x)
        test_y_pred, test_loss = SoftMax_CrossEnt_out.ForPass(test_y ,activation2)
            
        # find error  
        train_acc = Accur(train_y, train_y_pred)
        test_acc = Accur(test_y, test_y_pred)
        
        print(' Iteration %d. Accuracy is: %3f (train) and %3f (test)'% (i, train_acc, test_acc))
        trainAcc.append(train_acc) # save for plot
        testAcc.append(test_acc)
    
### Save model
    fileName = "ICA1part2_B_Final.ckpt" 
    Save_Model(fileName)
    
else:
### Load model
    fileName = "ICA1part2_B_Final.ckpt" 
    Load_Model(fileName)

### ------------------------------ RESULTS ------------------------------------
# Plot
make_plot2(trainAcc, "B", "  Train")
make_plot2(testAcc, "B", "  Test")

# find error as 1 - accuracy
# Final error 
fin_test_err = round(1 - test_acc,3)
fin_train_err =round(1 - train_acc,3)

print(' After %d epoch iterations, Final error is: %.3f (train) and %.3f (test)'% (max_iter, fin_train_err, fin_test_err))

# Confusion matrix
confusion = conf_mat(test_y_pred, test_y)

# Print final weight and bias terms:
print('\n After %d epoch iterations, Final weight array is:\n '% (max_iter))
print(np.round(w,3))
print('\n After %d epoch iterations, Final bias array is:\n ' % (max_iter))
print(np.round(b,3))

