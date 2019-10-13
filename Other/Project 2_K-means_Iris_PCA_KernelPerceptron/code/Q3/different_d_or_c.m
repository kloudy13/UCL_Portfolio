n=round((2/3)*length(y_train));
% training data
X = x_tr(1:n, :);
y=y_tr(1:n,:);
yb = y_train(1:n, :);
% testing (validation) data
Xt = x_tr(n+1:end, :);
ytb = y_train(n+1:end, :);
ytb_digit=y_tr(n+1:end,:);


%calculate the sum of errors- se for each d=2,3,..7
%type='gauss'
type='polynomial'
se=[];
;
for d=1:7 %for polynomial type kernel
%for d=0.01:0.01:0.06 %for gauss type kernel
[alpha] = perceptron_kernel(X, yb, d,20,type);

% Prediction part
% Initialise confidence and prediction vectors
y_pred = -ones(length(Xt), 10);
conf_levels = zeros(length(Xt), 1);

% Calculate kernel
K = kernel(X,Xt,d,type);

% predictions and confidence levels
for test= 1: length(Xt) 
    weights= 0;
    for train=1: length(X)
      weights = weights + alpha(train, :) * K(train, test);
    end
   [argmax, pos] = max(weights);
   conf_levels(test) = weights(pos);
   y_pred(test, pos) = 1;
end

error = sum([ytb ~= y_pred])/length(ytb)*100;
se=sum(error)
end
%poly: 6.0082, 5.6790, 5.0206, 5.5967, 6.4198, 5.5144=>best d=4
%gauss; (0.01,0.02,..0.07) 5.4321, 5.1852, 6.6667, 8.0658, 8.3128,
%9.5473=>best=0.02