function w_star = ridgesolution(y, X, lamda)
% Calculates w* according to ridge regression

w_star = (X'*X + lamda*size(X,1)*eye(size(X,2)))\ X'*y;