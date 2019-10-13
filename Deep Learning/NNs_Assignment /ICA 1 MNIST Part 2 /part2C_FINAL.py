# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 22:26:41 2017

@author: klaudia PART 2C
"""

# code description  
# In this part of teh essignment I implemnet part 1 without using tensor flow


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
        np.random.seed(1)
        self.name = "Linear Layer"
        self.w = sd * np.random.randn(dim1, dim2)  # model weights
        self.b = sd * np.random.randn(dim2)   # model bias
        self.x = None # data
        self.y_pred = None # predicted lables
        self.ddx = None 
        self.ddw = None
        self.ddb = None 

    def ForPass(self, x):
        y_pred = np.add(np.dot(x, self.w), self.b)
        self.y_pred = y_pred
        self.x = x
        return (self.y_pred)
        
    def WeightsOut(self):    
        return (self.w, self.b)

    def BackPass(self, ddy):
        self.ddx = np.dot(ddy, self.w.T) # take the transpose of w
        self.ddw = (self.x.T).dot(ddy)  
        self.ddb = ddy.sum(axis=0)
        return (self.ddx)

    def Gradient(self, ddy):
        self.ddw = (self.x.T).dot(ddy)
        self.ddb = ddy.sum(axis=0)
        return (self.ddw, self.ddb)

    def Update(self, rate): 
        self.w -= (rate / np.shape(self.x)[0]) * self.ddw
        self.b -= (rate / np.shape(self.x)[0]) * self.ddb
        return (self.w, self.b)

class CrossEntropy(object):
    
    def __init__(self):
        self.name = "Cross Entropy Loss"
    
    def ForPass(self, activ, y):
        out = np.mean(np.sum(-y * np.log(activ), axis=1))
        return (out)        
    
    def BackPass(self,probabl,y):
        return ((probabl - y)/probabl.shape[0])


class SoftMax():
    
    def __init__(self):
        self.name = "Softmax"
        
    def ForPass(self,y_pred):
        exp_y_pred = np.exp(y_pred)
        sum_exp_y_pred = np.sum(exp_y_pred, axis=1,keepdims=True)
        y_pred = exp_y_pred/ sum_exp_y_pred
        self.y_pred = y_pred   
        return (self.y_pred)        

    def BackPass(self,inp):
        return (inp)
                
        
class ReLU(object):
    def __init__(self):
        self.b_relu = None
        self.f_relu = None
    
    def ForPass(self, x):  # get_output
        self.f_relu = np.maximum(x, 0, x)
        return (self.f_relu)

    def BackPass(self,inp):
        b_relu = inp*(self.f_relu > 0).astype(float)
        return (b_relu)


# ---------------------------- OTHER FUNCTIONS -------------------------------- 
#calculate accuracy 
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


# =============================================================================
# SETUP to save/load
if Train == True:
    
# ========================= PART == 2C === SOLUTION ============================
#1 hidden layer (128 units) with a ReLU non-linearity, followed by a softmax

# ============================ Load dataset ===================================

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
    dim_batch = 100  # batch size
    max_iter = 50 # epochs
    
    # initialise lists to be stored
    trainAcc = []
    testAcc = []

## Define net:
    # Input layer
    dimH=128
    LinLayer1 =  LinLayer(dim_x, dimH) 
    HiddLayer = ReLU()
    LinLayer2 = LinLayer(dimH, dim_y)
    
    # Output layer
    SoftMax_out = SoftMax()
    #SoftMax_out = SoftMax_CrossEnt()
    Loss = CrossEntropy()
    
### TRAIN
    
    for i in range(max_iter):
        # SGD so need batching
        for batch in range(Train_dat_inst//dim_batch):
            batch_train_x = train_x[ batch * dim_batch : (batch+1) * dim_batch]
            batch_train_y = train_y[ batch * dim_batch : (batch+1) * dim_batch]

           # pass forward B
            lin1_out =  LinLayer1.ForPass(batch_train_x)
            activat =  HiddLayer.ForPass(lin1_out)
            lin2_out = LinLayer2.ForPass(activat)
            batch_train_y_pred = SoftMax_out.ForPass(lin2_out)
          
          # backward pass            
            bck = Loss.BackPass(batch_train_y_pred, batch_train_y)
            bck1 = SoftMax_out.BackPass(bck)
            bck2= LinLayer2.BackPass(bck1)
            bck3 = HiddLayer.BackPass(bck2)
            bck4 = LinLayer1.BackPass(bck3)
            
          # gradient descent
            LinLayer1.Update(rate)
            LinLayer2.Update(rate)
        # end batching
   
   ### caluclate results (accuracy)    
        # pass forward
        lin1_out =  LinLayer1.ForPass(train_x)
        activat =  HiddLayer.ForPass(lin1_out)
        lin2_out = LinLayer2.ForPass(activat)
        train_y_pred = SoftMax_out.ForPass(lin2_out)
        w ,b = LinLayer2.WeightsOut() # get final weights
        
        #test set
        lin1_out =  LinLayer1.ForPass(test_x)
        activat =  HiddLayer.ForPass(lin1_out)
        lin2_out = LinLayer2.ForPass(activat)
        test_y_pred = SoftMax_out.ForPass(lin2_out)
    
        # find error  
        train_acc = Accur(train_y, train_y_pred)
        test_acc = Accur(test_y, test_y_pred)
        
        print(' Iteration %d. Accuracy is: %3f (train) and %3f (test)'% (i, train_acc, test_acc))
        trainAcc.append(train_acc) # save for plot
        testAcc.append(test_acc)
 
 ### Save model
    fileName = "Part2_C.ckpt" 
    Save_Model(fileName)
    
else:
# load model
    fileName = "Part2_C.ckpt" 
    Load_Model(fileName)
            
                  
### ------------------------------ RESULTS ------------------------------------
# Plot
make_plot2(trainAcc, "C", "  Train")
make_plot2(testAcc, "C", "  Test")

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
