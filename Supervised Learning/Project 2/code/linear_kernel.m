function k = linear_kernel(x_i,x_j,v)
% Linear Ridge Regression Kernel
k = (x_i'*x_j);
disp('Linear Kernel matrix:')
size(k)



