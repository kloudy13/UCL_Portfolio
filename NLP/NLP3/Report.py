
# coding: utf-8

# ## REPORT
# #### Dataset preparation: 
# - Randomly shuffled the data. 
# - Split into training and development sets for generic experimentation. 
# - Estimated generalisation performance of production models by 5-fold cross validation. 
# - Trained final model on the full data set.
# 
# #### Word Embeddings: 
# -  Due to the small size of the corpus, to avoid over-fitting, we did not train embedding vectors but rather imported GloVe pre-trained embeddings vectors. 
# - Experimented with three different embedding dimensions (50-100-300), selected the one that resulted in highest performance (100-dimensional).
# 
# #### Pre-processing: 
# -	Improved tokenization (conservatively) until we were able to achieve a ~98% match of our vocabulary to the imported GloVe vectors. Visually inspected missing words for obvious mistakes, checked frequency of occurrence in the corpus and mostly replaced them with the OOV token (the embedding of which was obtained as the average – per dimension – of all words in our dictionary, i.e. effectively representing an uninformative prior). 
# -	Sorted the stories according to sentence size and created a vector that contained the length of each sentence (required for efficient RNN calculations)
# -	Bucketed stories according to size and split into batches according to the selected batch size.
# 
# #### Model: 
# -	Experimented with several different architectures, based on the set to sequence models introduced by [Vinyals et al, ICLR2016] and further developed by [Logeswaran et al, ICLR2017], supplemented with ideas for conditional encoding and sentence embedding attention by [Rocktaschel et al, ICLR2016]. 
# -	**Implemented model**: encoder-decoder model, with attention mechanisms at each stage and additional skip connections to improve trainability and performance.
# 
# *Model architecture:*
# - **a.	Sentence embedding:** Five RNNs (GRU) cells with shared state vectors and sentence by sentence attention mechanism.
# - **b.	Encoder:** RNNs (GRU) coupled with a sentence attention mechanism. The GRUs receive no input and their hidden state is a concatenation of the output of the previous hidden state and the attention readout. The process is repeated a number of times (read cycles) in order to achieve a representation that is feed-order invariant. The final state of the encoder initialises the hidden state of the decoder. 	
# - **c.	Decoder:** RNNs (GRU) with an affine layer which attends the decoder output and the sentence embeddings and produces the output. 
# - **d.	Skip connections:** We introduced skip connections between the sentence embedding (a weighted sum of the non-attentive and the attentive part) and the decoder (i.e. bypassing the encoder).
# 
# **N.B.**: Although the original works that our model is based upon used LSTMs, due to limited computational resources, we resorted to the more efficient GRUs. According to [Jozefowicz at al, 2015], both units have comparable performance andthe GRU’s often outperform LSTMs. For the same reason, in the affine decoder, we replaced the tanh activation function with ReLU.
# 
# #### Data augmentation: 
# - Fed models with all random permutations of sentence orderings, which increased robustness and literally eliminated over-fitting (calculated training set accuracy was consistently lower than dev-set accuracy) but came at a large computational penalty (huge increase in training time). Final model injects a random sample of a fraction of permutations (hyperparameter).
# 
# #### Other Considerations: 
# -  Implemented L2 regularization on the input and output layers (attentive sentence embedder and attentive skip encoder) with a common regularisation parameter. 
# -	Investigated model sizes between 60 thousand and 10 million parameters. Final model has: 212577 parameters (excluding the pre-trained word embeddings) and a vocabulary of 26837 words.
#    
#    *Also implemented but discarded:*
# -	Bi-directional RNNs (for both sentence and story embeddings)
# -	Weighted encoder-decoder 
# -	Drop-out (model proved to complex to properly implement drop out)
# 
# #### Model hyperparameters: 
# learning rate, batch size, training epochs, word embedding size, sentence embedding size, encoder & attention scorer size, decoder size, affine layer size, num of read cycles, data augmentation size (num. of injected permutations), L2 regulariser size.  Performed an informal, (and necessarily  limited) due to lack of computational resources parameter search (by varying one parameter at a time and observing the error in the dev set).
# 
# #### Optimisation:
# We used the ADAM optimizer with a learning rate of 0.001, a first momentum coefficient of 0.9 and a second momentum coefficient of 0.999. Regularisation strength was set to 0.1. 
# 
# #### Implementation: 
# All components were implemented using a highly flexible class structure. Hyper-parameter tuning was carried out by calling the various models from a command line interface and automatically saving the results to files for post processing. 
# Training was carried out using AWS CPU's. 
# 
# #### Differences between our model and the provided model (notebook):
# The baseline notebook model is considerably simple. It learns a word embedding from the training set, calculates the mean of each sentences’ word vectors (the sentence embedding) and predicts with a linear combination of the stacked sentence vectors (however, with some tuning, even this simple model could achieve dev-set accuracy above 50%).  Our model implements a state-of-the-art method (encoder-attention-decoder-affine layer model), additionally experimenting with skip connections and data augmentation. 
# However, we did not manage to get the full potential of our model nor replicate the results of the literature. The model proved difficult to build (and debug) and with the limited computational resources that we had access to, we were unable to tune it within the available time-frame. The maximum CV accuracy that we achieved with the full model was 54.6 which forced us to take a more conservative approach and *submit our report with a model that omits the decoding stage, achieved better performance (1.3% higher) and, being less complex, should generalise better*.
# 
# #### Error analysis and improvements:
# 
# We did not impose a consistency check, thus our model occasionally predicts multiple instances of the same label. Inspecting sentences with 4/5 classification accuracy we realised that this proves to be a considerable bottleneck. Given more time, we would have experimented with a different output set-up which would have allowed us to extract top-k most likely predictions which would have helped to alleviate that issue. 
# We also observed that the most common mistake that our model makes is to flip around two sentence positions resulting in 3/5 accuracy. Although this is reducing the accuracy of the model by two points in each story it is in essence a single mistake, and the model seems to correctly capture the general sentence ordering, which is satisfactory. 
# Lastly, by inspecting the errors, we did observe that some stories themselves have ambiguous true labels, for instance in story 5 of the dev set the order [0, 4, 2, 3, 1] is equally plausible as the ‘true’ order [1, 4, 2, 3, 0] and could therefore be miss-classified even by humans.
# Based on the gradual but constant increase in performance that we observed during the model tunning phase (and what we have read in the literature) we have strong reasons to believe that with more fine-tunning and longer training, the full model has the potential to achieve a much higher level of accuracy.  
# 
# ### References:
# 
# * L. Logeswaran,H.Lee, Dragomir Radev, “Sentence Ordering Using RNN”, ICLR, 2017
# * O. Vinyuals, S. Bengio, M. Kudlur, "Order matters: Sequence to sequence models for sets", ICLR 2016
# * T. Rocktaschel, E. Grefenstette, K.M. Hermann, T. Kocisky, P.Blunsom, "Reasoning with entailment with neural attention", ICLR 2016
# * Rafal Jozefowicz, Wojciech Zaremba, Ilya Sutskever,”Evolving Recurrent Neural Network Architectures”,JMLR 2015.
# * Wojciech Zaremba_Ilya Sutskever, Oriol Vinyals, , “Recurrent Neural Network Regularization”, ICLR 2015
# 
# ### Appendix:
# 
# ##### Encoder attention mechanism, from [Logeswaran et al, ICLR2017]:
# 
# $\quad \overline{ h_{enc}^{t}}, c_{enc}^{t} = LSTM( h_{enc}^{t}, c_{enc}^{t}) \\ \\
#     \quad e_{enc}^{t,i} = f(s_i, \overline{ h_{enc}^{t}}); i\in{1,...,n} \\ \\
#     \quad a_{enc}^{t} = Softmax( e_{enc}^{t})  \\ \\
#     \quad s_{att}^{t} = \sum  a_{enc}^{t,i}s_{i} \\ \\
#     \quad { h_{enc}^{t}} = [\overline{h_{enc}^{t}}, s_{att}^{t}]    \\ \\$
# 
# ** Where:**
# ${ h_{enc}^{t}}, c_{enc}^{t}$: hidden state of LSTM, $e_{enc}^{t,i}$:encoder output, $ s_{att}^{t}$: attention readout vector, $f()$: scoring function, $[\overline{h_{enc}^{t}}, s_{att}^{t}]$: concatenation of encoder hidden state and attention readout vector
# 
# ##### Sentence embedding attention mechanism,  from [Rocktaschel et al, ICLR2016]:
# 
# $ M_{t}= \tanh (W^{y}Y+ (W^{h}h_{t} + W^{r}r_{t-1})\otimes e_{L}) \quad  M_{t}\in\mathbb{R^{k\times L}}  \\$
#   
#   $\alpha_{t}= softmax(w^{T}M_{t}); \quad \alpha_{t}\in\mathbb{R^{L}} \\ \\$
#   
#   $ r_{t}=Y\alpha_{t}^{T}+\tanh(W^{t}r_{t-1});\quad r_{t}\in\mathbb{R^{k}}\\ \\$
#   
#   $ h^{*}=tanh(W^{p}r_{N}+W^{x}h_{N}; \quad h^{*}\in\mathbb{R^{k}}\\ $
#   
#  ** Where:**
#  $M_{t}$:attention matrix, $W^{y}$, $W^{h}$, $W^{r}$: trained projection matrices, $e_{L}\in\mathbb{R^L}$: unit vector,
#  $w^{k}\in\mathbb{R^{k}}$: trained parameter vector, $alpha$: attention weights, $r_{t}$: attention representation (corresponding to current sentence) 
# 
#  
