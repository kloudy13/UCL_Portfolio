function k = gauss_kernel(x_i,x_j,v)
% Gaussian Ridge Regression Kernel
% Inputs:
%           input vectors, x_i, x_j
%           kernel width, v
% [n,d] = size(x_i);
k = exp(-(norm(x_i-x_j,2)^2)/(2*v^2));
    % disp('Gauss Kernel matrix:');
    % size(k);


