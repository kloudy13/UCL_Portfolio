function Q4_minas
%% Q4 
clear all;
close all;
rng(0)

% Set Initial parameters
m = 100; 
d = 10;
v = 5;
e_amp = 1;
miou = 0;
sigma = 0.5;
split1 = 0.13;
split2 = 0.26;
split3 = 1;
size_train = split1*m;
size_valid = (split2-split1)*m;
size_test = (split3-split2)*m;


%% Part a
looplengtha = 100;
sigma_sq_v = (sigma^2)*logspace(-2,2,41)';
log2pi = log(2*pi);
loglik_a = zeros(looplengtha,size_train);
tmp = 0;
pos = 0;
atmp = size(size_train,d);
Ya_pred = size(size_test);
MSE_a = zeros(looplengtha,1);
best_lambda_a = zeros(length(sigma_sq_v),1);

% loop over 100 trials
for loop1 = 1:looplengtha;
%loop1 = 1;
    % Generate data
    [Ya,xa,ka] = GP(m,d,v,e_amp,miou,sigma);
    % Split the data
    xatrain = xa(1:split1*m,:);
    yatrain = Ya(1:split1*m,:);
    xavalid = xa(1+split1*m:split2*m,:);
    yavalid = Ya(1+split1*m:split2*m,:); 
    xatest = xa(1+split2*m:end,:);
    yatest = Ya(1+split2*m:end,:);
    Ka1 = zeros(size_train,size_train);
    
    I = eye(size_train);
    % loop over all values of a_sigma_sq
    for loop2 = 1:length(sigma_sq_v);
    %loop2 = 20;
        % initiate vectors
                
        % perform regression
        Ka1 = km_kernel(xatrain,xatrain,'gauss',sigma_sq_v(loop2));
      
        % calculate loglikelihood of data (evidence metric)
        loglik_a(loop1,loop2) = -0.5*yatrain'*((Ka1+sigma_sq_v(loop2)*I)\yatrain)...
                          -0.5*logDet((Ka1+sigma_sq_v(loop2)*I))-(m/2)*log2pi;
    end
    % select parameter based of max loglikelihood
    [~, pos] = max(loglik_a(loop1,:));
    % save lambda
    best_lambda_a(loop1) = sigma_sq_v(pos);
    % predict on the test data
    [atmp,Ya_pred] = km_krr(xatrain,yatrain,'gauss',v,best_lambda_a(loop1),xatest);
    % save MSE
    MSE_a(loop1) = MSE(yatest,Ya_pred);
end

%% Part b
looplengthb = 100;
looplengthb2 = length(sigma_sq_v);
Yb_pred = size(size_test);
apred_b = size(size_train,d);
Ypred_b = size(size_valid,1);
MSE_b = zeros(looplengthb,1);
MSE_loop_b = zeros(looplengthb2,1);
best_lambda_b = zeros(length(sigma_sq_v),1);

% loop over 100 trials
for loop1 = 1:looplengthb;
%loop1 = 1;
    % Generate data
    [Yb,xb,kb] = GP(m,d,v,e_amp,miou,sigma);
    % Split the data
    xbtrain = xb(1:split1*m,:);
    ybtrain = Yb([1:split1*m],:);
    xbvalid = xb(1+split1*m:split2*m,:);
    ybvalid = Yb([1+split1*m:split2*m],:); 
    xbtest = xb(1+split2*m:end,:);
    ybtest = Yb(1+split2*m:end,:);
    Kb1 = zeros(size_train,size_train);
    I = eye(size_train);
    
    % loop over all values of a_sigma_sq
    for loop2 = 1:looplengthb2;
    %loop2 = 20;
        % initiate vectors
                
        % perform regression
        [apred_b,Ypred_b] = km_krr(xbtrain,ybtrain,'gauss',v,sigma_sq_v(loop2),xbvalid);
        MSE_loop_b(loop2) = MSE(ybvalid,Ypred_b);
    end
    % select parameter based of least validation MSE
    [tmp pos] = min(MSE_loop_b);
    % save lambda
    best_lambda_b(loop1) = sigma_sq_v(pos);
    % predict on the test data
    [atmp,Yb_pred] = km_krr(xbtrain,ybtrain,'gauss',v,best_lambda_b(loop1),xbtest);
    % save MSE
    MSE_b(loop1) = MSE(ybtest,Yb_pred);
end
figure(1);
subplot(2,1,1);plot(MSE_a);
subplot(2,1,2);plot(MSE_b);
title('MSE - upper method a, lower method b');
figure(2);
subplot(2,1,1);hist(MSE_a,50);
subplot(2,1,2);hist(MSE_b,50);
title('Hist of MSE - upper method a, lower method b');
figure(3);
subplot(2,1,1);hist(best_lambda_a);
subplot(2,1,2);hist(best_lambda_b);
title('Hist best lambda - upper method a, lower method b');
figure(4);
subplot(2,1,1);plot(best_lambda_a);
subplot(2,1,2);plot(best_lambda_b);
title('Plot Best lambda - upper method a, lower method b');
figure(5);
plot(loglik_a(1,:));
hold on;
for i = 2:looplengtha;
    plot(loglik_a(i,:));
end
hold off;
title('Loglikelihood');
figure(6);
plot(ybtest,'g');
hold on;
plot(Ya_pred,'r');
plot(Yb_pred,'m');
hold off;
title('Yb test (g) vs yb_hat_a (r) vs yb_hat_c (m)');
disp(mean(best_lambda_a));
disp(mean(best_lambda_b));
disp(mean(MSE_a));
disp(mean(MSE_b));


%returns log of determinant of matrix 
%The determinant is a very small number 
%If we compute it and then take the log 
%we might have numerical underflow.  
function ld = logDet(A)

[U, L, V] = svd(A);
ld = sum(log(diag(L)));

