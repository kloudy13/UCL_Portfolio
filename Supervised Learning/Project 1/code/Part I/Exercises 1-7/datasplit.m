function [test, validation] = datasplit(X,testsize)
% Function splits data X in to a test set of size 'testSize' and the
% remaining into a  validation set.

test = X(1:testsize,:);
validation = X(testsize + 1:size(X,1), :);