
# coding: utf-8

# <!---
# Latex Macros
# -->
# $$
# \newcommand{\bar}{\,|\,}
# \newcommand{\Xs}{\mathcal{X}}
# \newcommand{\Ys}{\mathcal{Y}}
# \newcommand{\y}{\mathbf{y}}
# \newcommand{\weights}{\mathbf{w}}
# \newcommand{\balpha}{\boldsymbol{\alpha}}
# \newcommand{\bbeta}{\boldsymbol{\beta}}
# \newcommand{\aligns}{\mathbf{a}}
# \newcommand{\align}{a}
# \newcommand{\source}{\mathbf{s}}
# \newcommand{\target}{\mathbf{t}}
# \newcommand{\ssource}{s}
# \newcommand{\starget}{t}
# \newcommand{\repr}{\mathbf{f}}
# \newcommand{\repry}{\mathbf{g}}
# \newcommand{\x}{\mathbf{x}}
# \newcommand{\prob}{p}
# \newcommand{\vocab}{V}
# \newcommand{\params}{\boldsymbol{\theta}}
# \newcommand{\param}{\theta}
# \DeclareMathOperator{\perplexity}{PP}
# \DeclareMathOperator{\argmax}{argmax}
# \DeclareMathOperator{\argmin}{argmin}
# \newcommand{\train}{\mathcal{D}}
# \newcommand{\counts}[2]{\#_{#1}(#2) }
# \newcommand{\length}[1]{\text{length}(#1) }
# \newcommand{\indi}{\mathbb{I}}
# $$

# # Assignment 3

# ## Introduction
# 
# In the last assignment, you will apply deep learning methods to solve a particular story understanding problem. Automatic understanding of stories is an important task in natural language understanding [[1]](http://anthology.aclweb.org/D/D13/D13-1020.pdf). Specifically, you will develop a model that given a sequence of sentences learns to sort these sentence in order to yield a coherent story [[2]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/short-commonsense-stories.pdf). This sounds (and to an extent is) trivial for humans, however it is a quite difficult task for machines as it involves commonsense knowledge and temporal understanding.

# ## Goal
# 
# You are given a dataset of 45502 instances, each consisting of 5 sentences. Your system needs to ouput a sequence of numbers which represent the predicted order of these sentences. For example, given a story:
# 
#     He went to the store.
#     He found a lamp he liked.
#     He bought the lamp.
#     Jan decided to get a new lamp.
#     Jan's lamp broke.
# 
# your system needs to provide an answer in the following form:
# 
#     2	3	4	1	0
# 
# where the numbers correspond to the zero-based index of each sentence in the correctly ordered story. So "`2`" for "`He went to the store.`" means that this sentence should come 3rd in the correctly ordered target story. In This particular example, this order of indices corresponds to the following target story:
# 
#     Jan's lamp broke.
#     Jan decided to get a new lamp.
#     He went to the store.
#     He found a lamp he liked.
#     He bought the lamp.

# ## Resources
# 
# To develop your model(s), we provide a training and a development datasets. The test dataset will be held out, and we will use it to evaluate your models. The test set is coming from the same task distribution, and you don't need to expect drastic changes in it.
# 
# You will use [TensorFlow](https://www.tensorflow.org/) to build a deep learning model for the task. We provide a very crude system which solves the task with a low accuracy, and a set of additional functions you will have to use to save and load the model you create so that we can run it.
# 
# As we have to run the notebooks of each submission, and as deep learning models take long time to train, your notebook **NEEDS** to conform to the following requirements:
# * You **NEED** to run your parameter optimisation offline, and provide your final model saved by using the provided function
# * The maximum size of a zip file you can upload to moodle is 160MB. We will **NOT** allow submissions larger than that.
# * We do not have time to train your models from scratch! You **NEED** to provide the full code you used for the training of your model, but by all means you **CANNOT** call the training method in the notebook you will send to us.
# * We will run these notebooks automatically. If your notebook runs the training procedure, in addition to loading the model, and we need to edit your code to stop the training, you will be penalised with **-20 points**.
# * If you do not provide a pretrained model, and rely on training your model on our machines, you will get **0 points**.
# * It needs to be tested on the stat-nlp-book Docker setup to ensure that it does not have any dependencies outside of those that we provide. If your submission fails to adhere to this requirement, you will get **0 points**.
# 
# Running time and memory issues:
# * We have tested a possible solution on a mid-2014 MacBook Pro, and a few epochs of the model run in less than 3min. Thus it is possible to train a model on the data in reasonable time. However, be aware that you will need to run these models many times over, for a larger number of epochs (more elaborate models, trained on much larger datasets can train for weeks! However, this shouldn't be the case here.). If you find training times too long for your development cycle you can reduce the training set size. Once you have found a good solution you can increase the size again. Caveat: model parameters tuned on a smaller dataset may not be optimal for a larger training set.
# * In addition to this, as your submission is capped by size, feel free to experiment with different model sizes, numeric values of different precisions, filtering the vocabulary size, downscaling some vectors, etc.

# ## Hints
# 
# A non-exhaustive list of things you might want to give a try:
# - better tokenization
# - experiment with pre-trained word representations such as [word2vec](https://code.google.com/archive/p/word2vec/), or [GloVe](http://nlp.stanford.edu/projects/glove/). Be aware that these representations might take a lot of parameters in your model. Be sure you use only the words you expect in the training/dev set and account for OOV words. When saving the model parameters, pre-rained word embeddings can simply be used in the word embedding matrix of your model. As said, make sure that this word embedding matrix does not contain all of word2vec or GloVe. Your submission is limited, and we will not allow uploading nor using the whole representations set (up to 3GB!)
# - reduced sizes of word representations
# - bucketing and batching (our implementation is deliberately not a good one!)
#   - make sure to draw random batches from the data! (we do not provide this in our code!)
# - better models:
#   - stacked RNNs (see tf.nn.rnn_cell.MultiRNNCel
#   - bi-directional RNNs
#   - attention
#   - word-by-word attention
#   - conditional encoding
#   - get model inspirations from papers on nlp.stanford.edu/projects/snli/
#   - sequence-to-sequence encoder-decode architecture for producing the right ordering
# - better training procedure:
#   - different training algorithms
#   - dropout on the input and output embeddings (see tf.nn.dropout)
#   - L2 regularization (see tf.nn.l2_loss)
#   - gradient clipping (see tf.clip_by_value or tf.clip_by_norm)
# - model selection:
#   - early stopping
# - hyper-parameter optimization (e.g. random search or grid search (expensive!))
#     - initial learning rate
#     - dropout probability
#     - input and output size
#     - L2 regularization
#     - gradient clipping value
#     - batch size
#     - ...
# - post-processing
#   - for incorporating consistency constraints

# ## Setup Instructions
# It is important that this file is placed in the **correct directory**. It will not run otherwise. The correct directory is
# 
#     DIRECTORY_OF_YOUR_BOOK/assignments/2016/assignment3/problem/group_X/
#     
# where `DIRECTORY_OF_YOUR_BOOK` is a placeholder for the directory you downloaded the book to, and in `X` in `group_X` contains the number of your group.
# 
# After you placed it there, **rename the notebook file** to `group_X`.
# 
# The notebook is pre-set to save models in
# 
#     DIRECTORY_OF_YOUR_BOOK/assignments/2016/assignment3/problem/group_X/model/
# 
# Be sure not to tinker with that - we expect your submission to contain a `model` subdirectory with a single saved model! 
# The saving procedure might overwrite the latest save, or not. Make sure you understand what it does, and upload only a single model! (for more details check tf.train.Saver)

# ## General Instructions
# This notebook will be used by you to provide your solution, and by us to both assess your solution and enter your marks. It contains three types of sections:
# 
# 1. **Setup** Sections: these sections set up code and resources for assessment. **Do not edit, move nor copy these cells**.
# 2. **Assessment** Sections: these sections are used for both evaluating the output of your code, and for markers to enter their marks. **Do not edit, move, nor copy these cells**.
# 3. **Task** Sections: these sections require your solutions. They may contain stub code, and you are expected to edit this code. For free text answers simply edit the markdown field.  
# 
# **If you edit, move or copy any of the setup, assessments and mark cells, you will be penalised with -20 points**.
# 
# Note that you are free to **create additional notebook cells** within a task section. 
# 
# Please **do not share** this assignment nor the dataset publicly, by uploading it online, emailing it to friends etc.

# ## Submission Instructions
# 
# To submit your solution:
# 
# * Make sure that your solution is fully contained in this notebook. Make sure you do not use any additional files other than your saved model.
# * Make sure that your solution runs linearly from start to end (no execution hops). We will run your notebook in that order.
# * **Before you submit, make sure your submission is tested on the stat-nlp-book Docker setup to ensure that it does not have any dependencies outside of those that we provide. If your submission fails to adhere to this requirement, you will get 0 points**.
# * **If running your notebook produces a trivially fixable error that we spot, we will correct it and penalise you with -20 points. Otherwise you will get 0 points for that solution.**
# * **Rename this notebook to your `group_X`** (where `X` is the number of your group), and adhere to the directory structure requirements, if you have not already done so. ** Failure to do so will result in -1 point.**
# * Download the notebook in Jupyter via *File -> Download as -> Notebook (.ipynb)*.
# * Your submission should be a zip file containing the `group_X` directory, containing `group_X.ipynb` notebook, and the `model` directory with _____
# * Upload that file to the Moodle submission site.

# ## <font color='green'>Setup 1</font>: Load Libraries
# This cell loads libraries important for evaluation and assessment of your model. **Do not change, move or copy it.**

# In[1]:

get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\n%matplotlib inline\n#! SETUP 1 - DO NOT CHANGE, MOVE NOR COPY\nimport sys, os\n_snlp_book_dir = "../../../../../"\nsys.path.append(_snlp_book_dir)\n# docker image contains tensorflow 0.10.0rc0. We will support execution of only that version!\nimport statnlpbook.nn as nn\n\nimport tensorflow as tf\nimport numpy as np')


# ## <font color='green'>Setup 2</font>: Load Training Data
# 
# This cell loads the training data. **Do not edit the next cell, nor copy/duplicate it**. Instead refer to the variables in your own code, and slice and dice them as you see fit (but do not change their values). 
# For example, no one stops you from introducing, in the corresponding task section, `my_train` and `my_dev` variables that split the data into different folds.   

# In[2]:

#! SETUP 2 - DO NOT CHANGE, MOVE NOR COPY
import tensorflow as tf
import numpy as np

data_path = _snlp_book_dir + "data/nn/"
data_train = nn.load_corpus(data_path + "train.tsv")
data_dev = nn.load_corpus(data_path + "dev.tsv")
assert(len(data_train) == 45502)


# ### Data Structures
# 
# Notice that the data is loaded from tab-separated files. The files are easy to read, and we provide the loading functions that load it into a simple data structure. Feel free to check details of the loading.
# 
# The data structure at hand is an array of dictionaries, each containing a `story` and the `order` entry. `story` is a list of strings, and `order` is a list of integer indices:

# In[3]:

data_train[0]


# ## <font color='blue'>Task 1</font>: Model implementation
# 
# Your primary task in this assignment is to implement a model that produces the right order of the sentences in the dataset.
# 
# ### Preprocessing pipeline
# 
# First, we construct a preprocessing pipeline, in our case `pipeline` function which takes care of:
# - out-of-vocabulary words
# - building a vocabulary (on the train set), and applying the same unaltered vocabulary on other sets (dev and test)
# - making sure that the length of input is the same for the train and dev/test sets (for fixed-sized models)
# 
# You are free (and encouraged!) to do your own input processing function. Should you experiment with recurrent neural networks, you will find that you will need to do so.

# In[4]:

def tokenize(input):
    tokenized = re.compile('[\s.,]').split(input)
    return [token.lower() for token in tokenized]


# In[5]:

# convert train set to integer IDs
train_stories, train_orders, vocab = nn.pipeline(data_train)


# You need to make sure that the `pipeline` function returns the necessary data for your computational graph feed - the required inputs in this case, as we will call this function to process your dev and test data. If you do not make sure that the same pipeline applied to the train set is applied to other datasets, your model may not work with that data!

# In[6]:

# get the length of the longest sentence
max_sent_len = train_stories.shape[2]

# convert dev set to integer IDs, based on the train vocabulary and max_sent_len
dev_stories, dev_orders, _ = nn.pipeline(data_dev, vocab=vocab, max_sent_len_=max_sent_len)


# You can take a look at the result of the `pipeline` with the `show_data_instance` function to make sure that your data loaded correctly:

# In[7]:

nn.show_data_instance(dev_stories, dev_orders, vocab, 155)


# ### Model
# 
# The model we provide is a rudimentary, non-optimised model that essentially represents every word in a sentence with a fixed vector, sums these vectors up (per sentence) and puts a softmax at the end which aims to guess the order of sentences independently.
# 
# First we define the model parameters:

# In[8]:

### MODEL PARAMETERS ###
target_size = 5
vocab_size = len(vocab)
input_size = 40 # varry dimensionality 
# n = len(train_stories)
output_size = 5


# and then we define the model

# In[9]:

### MODEL ###

## PLACEHOLDERS
story = tf.placeholder(tf.int64, [None, None, None], "story")        # [batch_size x 5 x max_length]
order = tf.placeholder(tf.int64, [None, None], "order")              # [batch_size x 5]

batch_size = tf.shape(story)[0]

sentences = [tf.reshape(x, [batch_size, -1]) for x in tf.split(1, 5, story)]  # 5 times [batch_size x max_length]

# Word embeddings
initializer = tf.random_uniform_initializer(-0.1, 0.1)
embeddings = tf.get_variable("W", [vocab_size, input_size], initializer=initializer)

## we shoudl end with a better word embedig 

sentences_embedded = [tf.nn.embedding_lookup(embeddings, sentence)   # [batch_size x max_seq_length x input_size]
                      for sentence in sentences]

hs = [tf.reduce_sum(sentence, 1) for sentence in sentences_embedded] # 5 times [batch_size x input_size]

h = tf.concat(1, hs)    # [batch_size x 5*input_size]
h = tf.reshape(h, [batch_size, 5*input_size])

logits_flat = tf.contrib.layers.linear(h, 5 * target_size)    # [batch_size x 5*target_size]
logits = tf.reshape(logits_flat, [-1, 5, target_size])        # [batch_size x 5 x target_size]

###Lstm goes here ish 

# loss 
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, order))

# prediction function
unpacked_logits = [tensor for tensor in tf.unpack(logits, axis=1)]
softmaxes = [tf.nn.softmax(tensor) for tensor in unpacked_logits]
softmaxed_logits = tf.pack(softmaxes, axis=1)
predict = tf.arg_max(softmaxed_logits, 2)


# We built our model, together with the loss and the prediction function, all we are left with now is to build an optimiser on the loss:

# In[10]:

opt_op = tf.train.AdamOptimizer(0.01).minimize(loss)


# ### Model training 
# 
# We defined the preprocessing pipeline, set the model up, so we can finally train the model

# In[11]:

BATCH_SIZE = 25

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    n = train_stories.shape[0]

    for epoch in range(5):
        print('----- Epoch', epoch, '-----')
        total_loss = 0
        for i in range(n // BATCH_SIZE):
            inst_story = train_stories[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            inst_order = train_orders[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            feed_dict = {story: inst_story, order: inst_order}
            _, current_loss = sess.run([opt_op, loss], feed_dict=feed_dict)
            total_loss += current_loss

        print(' Train loss:', total_loss / n)

        train_feed_dict = {story: train_stories, order: train_orders}
        train_predicted = sess.run(predict, feed_dict=train_feed_dict)
        train_accuracy = nn.calculate_accuracy(train_orders, train_predicted)
        print(' Train accuracy:', train_accuracy)
        
        dev_feed_dict = {story: dev_stories, order: dev_orders}
        dev_predicted = sess.run(predict, feed_dict=dev_feed_dict)
        dev_accuracy = nn.calculate_accuracy(dev_orders, dev_predicted)
        print(' Dev accuracy:', dev_accuracy)

        
    
    nn.save_model(sess)


# ## <font color='red'>Assessment 1</font>: Assess Accuracy (50 pts) 
# 
# We assess how well your model performs on an unseen test set. We will look at the accuracy of the predicted sentence order, on sentence level, and will score them as followis:
# 
# * 0 - 20 pts: 45% <= accuracy < 50%, linear
# * 20 - 40 pts: 50% <= accuracy < 55
# * 40 - 70 pts 55 <= accuracy < Best Result, linear
# 
# The **linear** mapping maps any accuracy value between the lower and upper bound linearly to a score. For example, if your model's accuracy score is $acc=54.5\%$, then your score is $20 + 20\frac{acc-50}{55-50}$.
# 
# The *Best-Result* accuracy is the maximum of the best accuracy the course organiser achieved, and the submitted accuracies scores.  

# Change the following lines so that they construct the test set in the same way you constructed the dev set in the code above. We will insert the test set instead of the dev set here. **`test_feed_dict` variable must stay named the same**.

# In[12]:

# LOAD THE DATA
data_test = nn.load_corpus(data_path + "dev.tsv")
# make sure you process this with the same pipeline as you processed your dev set
test_stories, test_orders, _ = nn.pipeline(data_test, vocab=vocab, max_sent_len_=max_sent_len)

# THIS VARIABLE MUST BE NAMED `test_feed_dict`
test_feed_dict = {story: test_stories, order: test_orders}


# The following code loads your model, computes accuracy, and exports the result. **DO NOT** change this code.

# In[13]:

#! ASSESSMENT 1 - DO NOT CHANGE, MOVE NOR COPY
with tf.Session() as sess:
    # LOAD THE MODEL
    saver = tf.train.Saver()
    saver.restore(sess, './model/model.checkpoint')
    
    # RUN TEST SET EVALUATION
    dev_predicted = sess.run(predict, feed_dict=test_feed_dict)
    dev_accuracy = nn.calculate_accuracy(dev_orders, dev_predicted)

dev_accuracy


# ## <font color='orange'>Mark</font>:  Your solution to Task 1 is marked with ** __ points**. 
# ---

# ## <font color='blue'>Task 2</font>: Describe your Approach
# 
# Enter a 750 words max description of your approach **in this cell**.
# Make sure to provide:
# - an **error analysis** of the types of errors your system makes
# - compare your system with the model we provide, focus on differences and draw useful comparations between them
# 
# Should you need to include figures in your report, make sure they are Python-generated. For that, feel free to create new cells after this cell (before Assessment 2 cell). Link online images at your risk.

# ## <font color='red'>Assessment 2</font>: Assess Description (30 pts) 
# 
# We will mark the description along the following dimensions: 
# 
# * Clarity (10pts: very clear, 0pts: we can't figure out what you did, or you did nothing)
# * Creativity (10pts: we could not have come up with this, 0pts: Use only the provided model)
# * Substance (10pts: implemented complex state-of-the-art classifier, compared it to a simpler model, 0pts: Only use what is already there)

# ## <font color='orange'>Mark</font>:  Your solution to Task 2 is marked with ** __ points**.
# ---

# ## <font color='orange'>Final mark</font>: Your solution to Assignment 3 is marked with ** __points**. 
