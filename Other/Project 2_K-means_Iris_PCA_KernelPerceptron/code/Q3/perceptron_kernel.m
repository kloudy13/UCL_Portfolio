function [alpha] = perceptron_kernel(X, y, d,epoch,type)
% PERCEPTRON_KERNEL perceptron algorithm + kernel
% usage: [alpha] = perceptron_dual_kernel(X, y, d, epoch, type)
% ell: the number of training samples
% n: length of the input vector
  [ell n] = size(X);
% compute the Kernel matrix (using function kernel.m)
  K = kernel(X,X,d,type);
% the dual variables alpha 
  alpha = zeros(ell,10);
% start repeat for loop
  a=0.2
% set a small possitive quantity for updating the alpha by 
% setting zero if the prediction is correct and y ifelse
% we need a and a/2 for the logical operations for setting
% y in the case of not correct prediction
 for i= 1:epoch,
    for j = 1:ell,
        alpha(j,:) = alpha(j,:) + (y(j,:).*(a*((K(j,:)*(alpha))>0)-a/2)<0).*y(j,:);
      end
    end
end
 
 

