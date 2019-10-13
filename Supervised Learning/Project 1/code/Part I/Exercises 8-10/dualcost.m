function out = dualcost(K,y,alpha)
% function calculates MSE on dual kernel ridge regression using Eq. 15 from
% the assignmnet handout.
% Input:
% K is the kernel matrix - should be square!
% y is a vector of the dependant variable
% alpha is the vector of dual weights
% Output:
% Out - single value of the MSE

    
out = (K * alpha - y)' * (K * alpha - y) / size(K,1);
end