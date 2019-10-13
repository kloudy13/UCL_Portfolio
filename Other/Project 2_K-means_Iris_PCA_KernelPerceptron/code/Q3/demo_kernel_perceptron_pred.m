close all
clear all
% import the data
training_set = load('ziptrain.dat.gz');
test_set = load('ziptest.dat.gz');
% the y corresponds to the specific digits
% split y and x for the training and the test set
y_tr=training_set(:,1);
y_tst=test_set(:,1);
x_tr=training_set(:,[2:end]);
x_test=test_set(:,[2:end]);
% convert to -1,1
[digit] = unique(y_tr);
digit=digit';
y_train=(digit==y_tr);
y_train=double(y_train);
y_train(y_train==0)=-1;
[digit] = unique(y_tst);
digit=digit';
y_test=(digit==y_tst);
y_test=double(y_test);
y_test(y_test==0)=-1;
% split the training set into a train and a validation(test) set
n=round((2/3)*length(y_train));
% training data
X = x_tr(1:n, :);
y=y_tr(1:n,:);
yb = y_train(1:n, :);
% testing (validation) data
Xt = x_tr(n+1:end, :);
ytb = y_train(n+1:end, :);
ytb_digit=y_tr(n+1:end,:);
%d=0.02; %optimum for gauss kernel
d=1; %optimum for polynomial kernel
%type='gauss' %gauss type kernel
type='polynomial' %polynomial type kernel
[alpha] = perceptron_kernel(X, yb, d,20,type);
y_pred = -ones(length(Xt), 10);
conf_levels = zeros(length(Xt), 1);
K = kernel(X,Xt,d,type);
% predictions and confidence levels
for test= 1: length(Xt), 
    weights= zeros(1,1);
    for train=1: length(X),
      weights = weights + alpha(train, :) * K(train, test);
    end
   [argmax, pos] = max(weights);
   conf_levels(test) = weights(pos);
   y_pred(test, pos) = 1;
end
% calculate the error as it described in HW
error = sum([ytb ~= y_pred])/length(ytb)*100;
% convert -1, 1 to digit
 a=(y_pred==1);
 a=double(a);
 digit_pred=[];
for j=1:size(a,1),
    for i=1:size(a,2),
        if a(j,i)==1
            digit_pred(j)=i;
        end
    end
end
digit_pred=digit_pred-1;
% find and plot the 5 most uncertain results
s=5;
[d, pos] = sort(conf_levels);
ytb_unc= ytb_digit(pos(1:5));
unc_pred = digit_pred(pos(1:5));
Xt_unc = Xt(pos(1:5), :);
d=16
figure;
hold on
for uncertain = 1:s,
subplot(3,2,uncertain)
colormap(gray(255));
train_x_reshape = (reshape(Xt_unc(uncertain, :), [d, d]))';
plot_image = imshow(imresize(train_x_reshape, d), []);
title({['Real Digit=', num2str(ytb_unc(uncertain))], ['Predicted Digit= ', num2str(unc_pred(uncertain))]})
end
 axis square
 hold off
 ability_pr=confusionmat(ytb_digit,digit_pred);
 error
 