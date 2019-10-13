
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

# # Assignment 2
# 
# ## Introduction
# In this assignment you will build the first stage of a biomedical event extractor. Biomedical events are state changes of biomolecules. For example, if you have a protein and you add a phosphate (PO4) group to it, this is referred to as a phosphorylation event. Many papers in the biomedical literature mention such events. The grand goal of biomedical event extraction is to teach machines how to read this literature and produce structured representations of biomedical events that biomedical researchers can query effectively. This task has received considerable attention in the NLP literature, and is the topic of a biennial [shared task](http://2011.bionlp-st.org/). We will use the data from this task as starting point for this assignment.   
# 
# To illustrate biomedical event extraction, let us consider an example. From the sentence 
# 
# > **phosphorylation** of TRAF2 **inhibits** **binding** to the CD40 domain
# 
# we could extract the structure 
# 
# > Negative_Regulation(Phosphorylation(TRAF2), Binding(TRAF2, CD40)
# 
# and store it in a database. Someone can then query this database, for example, to figure out all ways to prevent binding of TRAF2 to CD40.
# 
# The task is often divided into two steps. First you need to find **trigger** words in the sentence that correspond to biomedical events, and determine their event type *label*. For example, in the above sentence "phosphorylation" is a trigger word for an event of type "Phosphorylation", "inhibits" a trigger word for a "Negative Regulation" event, and "binding" a trigger for a "Binding" event. Notice that sometimes the type labels are obvious, but often they are not. Also note that the label of a word could be "None". For example, the word "of" in the above sentence has the label "None".  
# 
# The second step requires the extractor to produce **argument relations** between event triggers and protein mentions or other event triggers. For example, in the above case the argument of "phosphorylation" is "TRAF2", and one argument of "inhibits" is "phosphorylation of TRAF2" whereas the other is "binding to the CD40 domain". In this assignment you **do not have to do this**. We will focus on the event trigger detection problem exclusively. 
# 
# ## Goal
# Your goal is to develop an event trigger labeler. This extractor is given a sentence and a candidate token. Both constitute the input $\x$. One such input could be: 
# 
# > $\x$: phosphorylation of TRAF2 **inhibits** binding to the CD40 domain
# 
# The goal is to predict the label $y$ of the candidate event trigger. In the above case the label would be $y=\text{Negative_Regulation}$. 
# 
# Some candidates may not refer to event triggers at all. For example:
# 
# > $\x$: phosphorylation **of** TRAF2 inhibits binding to the CD40 domain
# 
# In such cases the label is $y=\text{None}$.
# 
# ## Resources
# To develop your model you have access to:
# 
# * The data in `data/bionlp/train`. This data can be split into training and dev set (as done below), or used for cross-validation.
# * Helper code stored in the python module [bio.py](/edit/statnlpbook/bio.py).
# * Libraries on the [docker image](https://github.com/uclmr/stat-nlp-book/blob/python/Dockerfile) which contains everything in [this image](https://github.com/jupyter/docker-stacks/tree/master/scipy-notebook), including scikit-learn and tensorflow. 
# 
# As we have to run the notebooks of all students, and because writing efficient code is important, **your notebook should run in 5 minutes at most**, on your machine. Further comments:
# 
# * We have tested a possible solution on the Azure VMs and it ran in about 30s, so it is possible to train a reasonable model on the data in reasonable time. If you find training times too long for your development cycle you can reduce the training set size. Once you have found a good solution you can increase the size again. Caveat: model parameters tuned on a smaller dataset may not be optimal for a larger training set.
# 
# * Try to run your parameter optimisation offline, such that in your answer notebook the best parameters are already set and don't need to be searched. Include your optimisation code in the notebook, but don't call it at each notebook run.

# ## Hint
# While you do not need to predict the arguments of an event, it is important to understand how trigger labels relate to the syntactic and semantic arguments of the trigger word. Features that can capture this relation might help you in improving the result. Do inspect the data and try to get an understanding of it. That said, you don't have to be a biomedical expert to do well in this task. A few of the best results on the task were achieved by NLP researchers without any biomedical experience. They would, however, still inspect the data carefully.  

# ## Setup Instructions
# It is important that this file is placed in the **correct directory**. It will not run otherwise. The correct directory is
# 
#     DIRECTORY_OF_YOUR_BOOK/assignments/2016/assignment2/problem/
#     
# where `DIRECTORY_OF_YOUR_BOOK` is a placeholder for the directory you downloaded the book to. After you placed it there, **rename the file** to your UCL ID (of the form `ucxxxxx`). 

# ## General Instructions
# This notebook will be used by you to provide your solution, and by us to both assess your solution and enter your marks. It contains three types of sections:
# 
# 1. **Setup** Sections: these sections set up code and resources for assessment. **Do not edit, move nor copy these cells**.
# 2. **Assessment** Sections: these sections are used for both evaluating the output of your code, and for markers to enter their marks. **Do not edit, move, nor copy these cells**.
# 3. **Task** Sections: these sections require your solutions. They may contain stub code, and you are expected to edit this code. For free text answers simply edit the markdown field.  
# 
# **If you edit, move or copy any of the setup, assessments and mark cells, you will be penalised with -10 points**.
# 
# Note that you are free to **create additional notebook cells** within a task section. 
# 
# Please **do not share** this assignment publicly, by uploading it online, emailing it to friends etc. 

# ## Submission Instructions
# 
# To submit your solution:
# 
# * Make sure that your solution is fully contained in this notebook. 
# * Make sure that your solution runs linearly from start to end (no execution hops). We will run your notebook in that order.
# * **If running your notebook produces a trivially fixable error that we spot, we will correct it and penalise you with -10 points. Otherwise you will get 0 points for that solution.**
# * **Rename this notebook to your UCL ID** (of the form "ucxxxxx"), if you have not already done so. ** Failure to do so will result in -1 point.**
# * Download the notebook in Jupyter via *File -> Download as -> Notebook (.ipynb)*.
# * Upload the notebook to the Moodle submission site.

# ## <font color='green'>Setup 1</font>: Load Libraries
# This cell loads libraries important for evaluation and assessment of your model. **Do not change, move or copy it.**

# In[81]:

get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\n%matplotlib inline\n#! SETUP 1 - DO NOT CHANGE, MOVE NOR COPY\nimport sys, os\n_snlp_book_dir = "../../../../"\nsys.path.append(_snlp_book_dir) \nimport math\nfrom collections import defaultdict\nimport statnlpbook.bio as bio')


# ## <font color='green'>Setup 2</font>: Load Training Data
# 
# This cell loads the training data. **Do not edit this setup section, nor copy it**. Instead refer to the variables in your own code, and slice and dice them as you see fit (but do not change their values). For example, no one will stop you from introducing, in the corresponding task section, `my_event_train` and `my_event_dev` variables that split the data into different folds.   
# 
# Notice that the data is loaded from `json` files like [this one](/edit/data/bionlp/train/PMC-1310901-00-TIAB.json). Generally, you do not need to understand this format, as we provide loading functions that produce more convenient data structures shown below. But do feel free to investigate. 

# In[82]:

#! SETUP 2 - DO NOT CHANGE, MOVE NOR COPY
import sys, os
_snlp_book_dir = "../../../../"
sys.path.append(_snlp_book_dir) 
import math
from collections import defaultdict
import statnlpbook.bio as bio

# ## Data Structures
train_path = _snlp_book_dir + "data/bionlp/train"
event_corpus = bio.load_assignment2_training_data(train_path)
event_train = event_corpus[:len(event_corpus)//4 * 3]
event_dev = event_corpus[len(event_corpus)//4 * 3:]
assert(len(event_train)==53988)
print(len(event_dev))

# The data comes in the form of pairs consisting of `EventCandidate` objects and their trigger labels. The `EventCandidate` class can be found in [bio.py](/edit/statnlpbook/bio.py).

# In[83]:

event_candidate, label = event_corpus[0]


# Event candidate objects specify the classification problem. They consist of a sentence `sent` and the position `trigger_index` of the trigger candidate word. 

# In[84]:

event_candidate.sent


# In[85]:

event_candidate.trigger_index


# Event candidates also have a set of candidate arguments. These point to token spans (index of first token, inclusive, index of last token, exclusive) in the sentence that may or may not be *arguments* of the event. In the full event extraction task one needs to predict which of these candidates are true arguments of the events. However, here we will ignore this task, and give you only the information what candidates exist, not what their labels are. Note that this information can still be **very important** to understand what type of event the candidate corresponds to, if any. 

# In[86]:

event_candidate.argument_candidate_spans[:4]


# You can compactly visualise the complete candidate using `bio.render_event`, as shown below. Here the green span corresponds to the token at the trigger index. The spans in red brackets correspond to the argument candidates. The blue spans are protein mentions. 

# In[87]:

bio.render_event(event_candidate)


# ### Sentences
# The sentence object of an event candidate provides additional information about the sentence, such as what spans are proteins, what Part-of-Speech labels the tokens have, and a dependency parse of the sentence. First, the `tokens` field of a sentence provides useful features of tokens: 

# In[88]:

event_candidate.sent.tokens[3]


# The `dependencies` field stores lexical dependencies between words:

# In[89]:

index=event_candidate.trigger_index
event_candidate.sent.dependencies[index]['head']


# You can render the dependency graph of a sentence like so:

# In[90]:

bio.render_dependencies(event_candidate.sent)


# You can learn about the dependency labels in the [Stanford typed dependencies manual](http://nlp.stanford.edu/software/dependencies_manual.pdf). We also provide [lecture notes on dependency parsing](/notebooks/chapters/Transition-based%20dependency%20parsing.ipynb), including various pointers to more information. 
# 
# The `mentions` stores which spans correspond to proteins.

# In[91]:

event_candidate.sent.mentions
#event_candidate.sent.events


# There are some convenience functions for the sentence to check all the syntactic parents or children of a token, or whether a specific token is within a protein mention. These can be useful when designing features:

# In[92]:

event_candidate.sent.parents[0], event_candidate.sent.children[0] 


# In[93]:

event_candidate.sent.is_protein[3], event_candidate.sent.is_protein[7]


# ### Labels
# It is useful to know the complete set of event labels:

# In[94]:

{y for _,y in event_corpus}


# ## <font color='blue'>Task 1</font>: Create a Feature Function
# 
# In this task you will extract a specific feature representation $\repr(\x)$ for an event candidate $\x$. In particular, we want to add as features the syntactic children (modifiers) of the trigger token, together with their syntactic dependency label. A modifier of a token $h$ is a token $m$ that modifies $h$'s meaning. For example, in the noun phrase "green light" the adjective "green" modifies the noun "light". We will refer to the modifier token as the "child", and the modified token as "parent". Correspondingly, in the dependency graph modifiers are the child nodes of the modified tokens.  
# 
# The feature function will have to be implemented as a python function that populates a python dictionary with key-value pairs where the key indicates both the word and syntactic label of the child. 
# 
# For example, consider the following event and dependency parse:

# In[95]:

example = event_corpus[398][0]
bio.render_dependencies(example.sent)


# Here the goal is to produce a dictionary that maps the strings "Child: det->The" and "Child: nn->PCR" to 1.0. 
# 
# To solve this task, implement the feature function below. The passed in `result` is a dictionary you need to populate with more entries, and the `event` argument indicates for which event you need to extract the features. We have already populated the function with some initial code that should get you started.  

# In[96]:

def add_dependency_child_feats(result, event):
    """
    Append to the `result` dictionary features based on the syntactic dependencies of the event trigger word of
    `event`. The feature keys should have the form "Child: [label]->[word]" where "[label]" is the syntactic label
    of the syntatic child (e.g. "det" in the case above), and "[word]" is the word of the syntactic child (e.g. "The" 
    in the case above).
    Args:
        result: a defaultdict that returns `0.0` by default. 
        event: the event for which we want to populate the `result` dictionary with dependency features.
    Returns:
        Nothing, but populates the `result` dictionary. 
    """
    index = event.trigger_index
    
    for child,label in event.sent.children[index]:
        result["Child: " + label + "->" + event.sent.tokens[child]['word']] += 1.0 


# ## <font color='red'>Assessment 1</font>: Test Feature Function (20 pts)
# Here we test whether your feature function populates the given dictionary correctly. If the result passes all three tests you get 10 pts. Of course, solutions that just manually populate the result with the specific key value pairs tested below will receive 0 pts as well.

# In[97]:

#! ASSESSMENT 1 - DO NOT CHANGE, MOVE NOR COPY
result = defaultdict(float)
add_dependency_child_feats(result, example)

check_1 = len(result) == 2
check_2 = result['Child: det->The'] == 1.0
check_3 = result['Child: nn->PCR'] == 1.0
(check_1, check_2, check_3)


# ## <font color='orange'>Mark</font>:  Your solution to Task 1 is marked with ** __ points**. 
# ---

# ## <font color='blue'>Task 2</font>: Implement Model
# 
# You are to implement the `predict_event_labels` function below. This function gets as input a list of event candidate objects, and then returns a sequence of corresponding labels. You can implement this function in any way you like, again utilising any library on the docker image. We have populated the cell and function with a simple implementation that uses the scikit-learn logistic regression model. You can use this as a starting point and focus on implementing better feature functions. You can also start from scratch if you like. 
# 

# In[142]:

from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

# converts labels into integers, and vice versa, needed by scikit-learn.
label_encoder = LabelEncoder()

# encodes feature dictionaries as numpy vectors, needed by scikit-learn.
vectorizer = DictVectorizer()


def event_feat(event):
    """
    This feature function returns a dictionary representation of the event candidate. 
    Features are entirely self contained within this function and not defined in separate functions.

    Args:
        event: the `EventCandidate` object to produce a feature dictionary for.
    Returns:
        a dictionary with feature keys/indices mapped to feature counts.
    """
    ####### SETUP:
    result = defaultdict(float)
    myprotein=''
    index = event.trigger_index

    
    ####### Trigger features
    
    #  What is the trigger word stem? and what part of speech does it represent 
    result['trigger_word=' + event.sent.tokens[event.trigger_index]['stem']] += 1.0
    result['trigger_word_POS=' + event.sent.tokens[event.trigger_index]['pos']] += 1.0
    
    
    # What are the children and parents of the trigger?
    for parent,label in event.sent.parents[index]:
        result["Parent: " + label + "->" + event.sent.tokens[parent]['word']+"Parent pos:"+ event.sent.tokens[parent]['pos']] += 1.0 
        
    for child,label in event.sent.children[index]:
        result["Children: " + label + "->" + event.sent.tokens[child]['word']+"Children pos:"+ event.sent.tokens[child]['pos']] += 1.0 
    
    # How many parents and children does the trigger have?
    result['Num of children: '] += len(event.sent.children[index])
    result['Num of parents: '] += len(event.sent.parents[index])
    
    # How long is trigger word?
    result['trigger length: ']+ len(event.sent.tokens[index]['word'])
    # parameters wich add a value greater than '1' to feature will be scaled by the wieghting 
    # of the LR model implemented below. Thus, this is aceptable and results are not normalised.
    
    
    ###### After examination, Realise proteins are another crucial element to focus on:
    #thus, the following features focus on proteins
    
    # Is the trigger a protein?
    if event.sent.is_protein[index]:
         result['protein'] +=1
    else:
        result['not_protein'] +=0
         
    # What protein is mentioned? (effectively also counting proteins again)
    for mentions in event.sent.mentions:
        s = int(mentions['begin'])
        f = int(mentions['end'])
        for k in range(s,f):
            myprotein=myprotein + event.sent.tokens[k]['word']
        result['mypotein: '+ myprotein] +=1
        # the above was doen as I attempted to implement a feture which counts protein 
        # repetitions, against intuition this did not improve the F1 score, but the above did.
        
    # How many mentions of proteins are there? (assign more weight if a lot of mentions of
    # protein in the event candidate.
    # Agian, scaling of this parameter will be taken care of by the logic regression model) 
    result['Num of mentions:' ] += len(event.sent.mentions) 
    
    # How many children does the protein have?
    childprot=[]
    for child,label in event.sent.children[index]: #check if protein and if true then count
            if event.sent.is_protein[child] == True:
                childprot.append(event.sent.is_protein[child])
    result["ChildrenProtein: "] += len(childprot)
    
  
    #### Other features -shift focus away from trigger 

    result['Num of event candidates excl. trigger'] +=len(event.argument_candidate_spans)
    
    # Numerous other fetures were tested but they detracted from F1 result. 
    
    return result


# Convert the event candidates and their labels into vectors and integers, respectively.
train_event_x = vectorizer.fit_transform([event_feat(x) for x,_ in event_train])
train_event_y = label_encoder.fit_transform([y for _,y in event_train])

# Create and train the model. Use balanced verion of Logistic regression with inverse 
# regularisation strength parameter set to C=1.25 (optimal performance)

lr = LogisticRegression(C=1.25,class_weight = 'balanced')
lr.fit(train_event_x, train_event_y) 

def predict_event_labels(event_candidates): #unchanged
    """
    This function receives a list of `bio.EventCandidate` objects and predicts their labels. 
    It has been modified from the template provided in an attempt to down-weight probabilities 
    assigned to the 'none' category. 
    
    Args:
        event_candidates: A list of `EventCandidate` objects to label.
    Returns:
        a list of event labels, where the i-th label belongs to the i-th event candidate in the input.
    """
    p=0.25 #threshold
    
    event_x = vectorizer.transform([event_feat(e) for e in event_candidates])
    
    prob = lr.predict_proba(event_x)
    
    for i in range(len(prob)):
        maxindex = (prob[i]).argmax()
        # below condition only valid if predicting a none label below paramete p threshold
        if maxindex == 4 and prob[i][4]<=p:  #4 is the index of None
            prob[i][4]=0 # effectively discard the predition and event_y will be the next best alternative
    
    event_y = label_encoder.inverse_transform([proba.argmax() for proba in prob]) 

    return event_y


# It is useful to inspect the performance of your model, and see where it makes errors, both on the training set (to check for underfitting) and the development set. We have provided you with utility functions to help with this inspection. Note that you don't have to use these utilities, or the cells below, but it can help you to improve your model, and also with the error analysis and description of the approach in Task 3. 
# 
# First, we give you a breakdown of precision, recall and F1 on different event types:

# In[143]:

# This line calls your function to produce labels for the test set
event_dev_guess = predict_event_labels([x for x,_ in event_dev[:]])

# This line produces a confusion matrix
cm_dev = bio.create_confusion_matrix(event_dev,event_dev_guess) 

# This line turns the confusion matrix into a evaluation table with Precision, Recall and F1 for all labels.
bio.full_evaluation_table(cm_dev)


# It is useful to inspect [bio.py](/edit/statnlpbook/bio.py) to see how we define precision, recall and F1 score in this context.
# 
# You can also display a confusion matrix to identify what types of errors you are currently making. Notice that the matrix ignores the "None"-"None" cell as its counts would overpower all other counts (try removing the `outside_label` argument). 

# In[100]:

import statnlpbook.util as util
util.plot_confusion_matrix_dict(cm_dev,90, outside_label="None")


# The confusion matrix can give you hints on what type of errors you should look for and improve upon. This macro view on your model's performance is often more powerful when combined with a micro view on the instances that produce these errors. You can find errors of a specific type using `bio.find_errors` as shown below:

# In[101]:

errors = bio.find_errors("Regulation","Positive_regulation", event_dev, event_dev_guess)[:3]
errors


# These errors you can then inspect in detail via `show_event_error`:

# In[102]:

bio.show_event_error(*errors[0])


# It can also be very useful to inspect your feature map for the given instance. Sometimes this leads you to find out that you have a bug in your feature calculation, or that the feature representation is still insufficient for other reasons.

# In[104]:

event_feat(errors[0][0])


# ## <font color='red'>Assessment 2</font>: Assess Accuracy (50 pts) 
# 
# We assess how well your model performs on some unseen test set. We will look at the F1 across all event types, and will score them as follows:
# 
# * 0-40pts: 17% <= F1 < 60%, linear
# * 40-50pts: 60% <= F1 < Best Result, linear
# 
# The **linear** mapping maps any F1 value between the lower and upper bound linearly to a score. For example, if your model's F1 score is $F=55$, then your score is $40\frac{F-17}{60-17}$. 
# 
# The *Best-Result* perplexity is the maximum of the best perplexity the course organiser achieved, and the submitted F1 scores.  

# In[105]:

#! ASSESSMENT 2 - DO NOT CHANGE, MOVE NOR COPY
_snlp_event_test = event_dev # This line will be changed by us after submission to point to a test set.
_snlp_event_test_guess = predict_event_labels([x for x,_ in _snlp_event_test[:]])
_snlp_cm_test = bio.create_confusion_matrix(_snlp_event_test,_snlp_event_test_guess)  
bio.evaluate(_snlp_cm_test)[2] # This is the F1 score


# ## <font color='orange'>Mark</font>:  Your solution to Task 2 is marked with ** __ points**. 
# ---

# ## <font color='blue'>Task 3</font>: Describe your Approach
# 
# In this assignment we were tasked with developing a biomedical event extractor, focusing on designing a set of features assigning weights to observations. These features were then fed to a logistic regression (LR) classification model to predict labels in 10 categories. SVM could be applied instead of LR, however results are largely similar. Other models, like Decision Trees or Neural Networks, could be attempted but would require more data and wouldn’t provide opportunity for feature engineering. The models performance was assessed using precision and recall metrics, combined into the F1 score.
# 
# Data exploration aided the selection of optimal features and model parameters. Almost 90% of all event candidates were labelled as ‘None’ and the least frequent  labels accounted for only fractions of a percent of the data (Protein-Catabolism 0.14%). Therefore, a ‘balanced’ version of the LR model was used, where weights are inversely proportional to the frequency of label occurrence. The regularisation strength parameter (C) was tuned and set to and optimal 1.25. This yelled a significant improvement in F1 as compared with the default LR implementation. Candidate arguments were also explored statistically, looking at the presence of stop-words, most common words and their stems. Term Frequency–Inverse Document Frequency statistics were also calculated. The stop-word ‘of’ was used as a trigger word many times and categorised as ‘none’ in almost all cases. For each label category, characteristic words were identified and features were designed to add additional weights to these. However, this did not improve the F1 score, signifying this was done implicitly by the LR model. 
# 
# Numerous feature functions have been trailed before proposing the final solution; features related to trigger words and proteins had the highest impact on model performance. Features quantifying the trigger: word, length, POS children and parents were implemented. For proteins; amount and type were counted together with the children they had. The number of protein parents detracted from the F1 score.The overall amount of event candidate arguments also improved predictions, even if not all were ‘true’ argument candidates. Contrary to expectations, features focusing on the presence, position and POS of the Head reduced F1.
# 
# Error Analysis :
# From the confusion matrix below; my model performed worst for predicting the Regulation label (F1=48%) closely followed by other regulation-related labels (Negative-regulation F1=56%, Positive-regulation F1=58%). Despite the relatively large amount of training data as compared to other classes, these three labels relate to ‘regulation’ and share similarities. Features aimed at assigning extra weight to specific rare words in these classes did not improve results.
# 
# The model would perform better if less data was classified as ‘None’. To improve this, ‘of’ and similar words shouldn’t be flagged as triggers but changing this is outside of the assignment’s scope. Instead, I attempted to down-weight the probabilities assigned to the 'None' label by the LR model using thresholding. Further tuning probabilities on all 10 classes could improve F1 but I run out of time to implement this.
# 

# In[144]:

#FIGURE: Breakdown of error

event_dev_guess = predict_event_labels([x for x,_ in event_dev[:]])
cm_dev = bio.create_confusion_matrix(event_dev,event_dev_guess) 
bio.full_evaluation_table(cm_dev)


# ## <font color='red'>Assessment 3</font>: Assess Description (30 pts) 
# 
# We will mark the description along the following dimensions: 
# 
# * Clarity (10pts: very clear, 0pts: we can't figure out what you did)
# * Creativity (10pts: we could not have come up with this, 0pts: Use only word based features of the trigger word)
# * Substance (10pts: implemented complex state-of-the-art classifier, 0pts: Only use what is already there)

# ## <font color='orange'>Mark</font>:  Your solution to Task 3 is marked with ** __ points**.
# ---

# ## <font color='orange'>Final mark</font>: Your solution to Assignment 2 is marked with ** __points**. 
