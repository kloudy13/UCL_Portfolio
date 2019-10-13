function w_star = estimate_w(y,X)
% function calulates w based on the matrix x and 

w_star = (X'*X)\ X'*y;

