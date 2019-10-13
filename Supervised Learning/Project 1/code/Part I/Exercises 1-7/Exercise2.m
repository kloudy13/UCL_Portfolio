%% Exercise 2

% ===== Run each section separately and in order ======

% I have reused much of the code from the previous exercise. The only real
% changes I have made are changing the input dimensions in the
% 'randomdata.m' function.

%% Section 1: part a
clear

solution = randomdata(600,10,100);

% unpack results:
Xtrain = solution.Xtrain;
ytrain = solution.ytrain;
Xtest = solution.Xtest;
ytest = solution.ytest;

%% Section 2: part b - (run part a first)

%=========================================================================
% ---- Repeat part b from exercise 1--------------------------------------
%=========================================================================

w_star = estimate_w(ytrain, Xtrain);

% use function 'myMSE' to calcuate the MSE.

MSE_train_b = myMSE(ytrain, Xtrain,w_star);
MSE_test_b = myMSE(ytest, Xtest,w_star);

% ========================================================================
% -----Repeat part c from exercise 1--------------------------------------
% ========================================================================

% reassign the train and test sets:

solution = randomdata(510,10,10);

% unpack results:
Xtrain = solution.Xtrain;
ytrain = solution.ytrain;
Xtest = solution.Xtest;
ytest = solution.ytest;

% use result from part a and function 'estimate_w' to find w.

w_star = estimate_w(ytrain, Xtrain);

% use function 'myMSE' to calcuate the MSE.

MSE_train_c = myMSE(ytrain, Xtrain,w_star);
MSE_test_c = myMSE(ytest, Xtest,w_star);

% ========================================================================
% ----- Repeat part d from exercise 1 ------------------------------------
% ========================================================================
run = 200;

%initialise MSE '_100' means training set of size 100. 
MSE_train_d_100 = zeros(run,1);
MSE_test_d_100 = zeros(run,1);

for i = 1:run
    %create data
    solution = randomdata(600,10,100);
    Xtrain = solution.Xtrain;
    ytrain = solution.ytrain;
    Xtest = solution.Xtest;
    ytest = solution.ytest;
    
    %find w_star
    w_star = estimate_w(ytrain, Xtrain);
    
    % find total MSE
    MSE_train_d_100(i) = myMSE(ytrain, Xtrain,w_star);
    MSE_test_d_100(i) = myMSE(ytest, Xtest,w_star);
end

%find average:
Avg_MSE_train_100 = mean(MSE_train_d_100);
Avg_test_100 = mean(MSE_test_d_100);

% repeat for training set of size 10

% Copy paste of above - just changed the randomdata function and
% relevant MSE variable names.

% initialise MSE, '_10' means training set of size 10
MSE_train_d_10 = zeros(run,1);
MSE_test_d_10 = zeros(run,1);

for i = 1:run
    %create data
    solution = randomdata(510,10,10);
    Xtrain = solution.Xtrain;
    ytrain = solution.ytrain;
    Xtest = solution.Xtest;
    ytest = solution.ytest;
    
    %find w_star
    w_star = estimate_w(ytrain, Xtrain);
    
    % find total MSE
    MSE_train_d_10(i) = myMSE(ytrain, Xtrain,w_star);
    MSE_test_d_10(i) = myMSE(ytest, Xtest,w_star);
end

%find average:
Avg_MSE_train_10 = mean(MSE_train_d_10);
Avg_test__10 = mean(MSE_test_d_10);

%% Section 5: Visualise Results (Not part of the Question)
close all
% Visualise Results:

runx = 1:run;
plot(runx, log(MSE_train_d_10), 'o')
hold on
plot(runx, log(MSE_train_d_100), '^')
title('Training MSE of Sizes 10 & 100 Across 200 Iteration')
xlabel('Iteration')
ylabel('log(MSE)')
legend('Training Size: 10', 'Training Size: 100' )

figure
plot(runx, log(MSE_test_d_10), 'o')
hold on
plot(runx, log(MSE_test_d_100), '^')
title('Test MSE of Sizes 10 & 100 Across 200 Iteration')
xlabel('Iteration')
ylabel('log(MSE)')
legend('Training Size: 10', 'Training Size: 100' )







