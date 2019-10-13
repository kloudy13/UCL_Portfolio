function mse = MSE(y,y_hat)
% Calulcate the MSE of two vectors
% Input:
%        y: true values
%        y_hat: calculated values
%m = size(y);
mse = mean((y-y_hat).^2);