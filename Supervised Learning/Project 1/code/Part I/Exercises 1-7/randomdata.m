function [output] = randomdata(data_size, dimension, trainSetSize)
% function generates random noisy data from standard normal distribution,
% splits data into training and test data.


% create a true weight vector:
w_true = randn(dimension, 1);

% Generate noisy sample of size 'size' and dimension 'dimension'
X = randn(data_size, dimension);
noise = randn(data_size, 1);

y = X*w_true + noise;

% split data into training set size: 'trainSetSize'
X_train = X(1:trainSetSize,:);
y_train = y(1:trainSetSize,:);

% create test set of size 'Size' - 'trainSetSize'
X_test = X(trainSetSize + 1:size(X,1),:);
y_test = y(trainSetSize + 1:size(y,1),:);

% pack results into buckets
output.Xtrain = X_train;
output.ytrain = y_train;
output.Xtest = X_test;
output.ytest = y_test;
output.wtrue = w_true;









