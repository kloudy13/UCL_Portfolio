function k = quadratic_kernel(x_i,x_j,v)
% Quadratic Ridge Regression Kernel
[m,d] = size(x_i);
k = reshape((x_i'*x_i),[d^2,1])'*reshape((x_j'*x_j),[d^2,1]);
%disp('Quadratic Kernel matrix:');
%size(k);



