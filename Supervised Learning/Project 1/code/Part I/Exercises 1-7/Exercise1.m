%% Exercise 1


% Experimentally investigate the effect of the training set size, making
% plots wherver this helps your understanding.

%================== RUN EACH SECTION IN ORDER=======================



%% Section 1: Part a
clear

% The function 'randomdata' returns requred results

solution = randomdata(600,1,100);

% unpack results:
Xtrain = solution.Xtrain;
ytrain = solution.ytrain;
Xtest = solution.Xtest;
ytest = solution.ytest;

%% Section 2: Part b
% Run part a first.

% use result from part a and function 'estimate_w.m' to find w.

w_star = estimate_w(ytrain, Xtrain);

% use function 'myMSE' to calcuate the MSE.

MSE_train_b = myMSE(ytrain, Xtrain,w_star);
MSE_test_b = myMSE(ytest, Xtest,w_star);

% print
fprintf('Training MSE: %d \n', MSE_train_b)
fprintf('Test MSE: %d \n', MSE_test_b)

%% Section 3: Part c
clear

% reassign the train and test sets. 

solution = randomdata(510,1,10);

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

% print
fprintf('Training MSE: %d \n', MSE_train_c)
fprintf('Test MSE: %d \n', MSE_test_c)

%% Section 4: Part d
clear 

run = 200;
%initialise MSE '_100' means training set of size 100. 

% Repeating part (a),(b) 200x:

MSE_train_d_100 = zeros(run,1);
MSE_test_d_100 = zeros(run,1);

for i = 1:run
    %create data
    solution = randomdata(600,1,100);
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
Avg_Train_MSE_100 = mean(MSE_train_d_100);
Avg_Test_MSE_100 = mean(MSE_test_d_100);

% Repeat part (c) 200x

% Copy paste of above - just changed the randomdata function and MSE
% variable names.

MSE_train_d_10 = zeros(run,1);
MSE_test_d_10 = zeros(run,1);

for i = 1:run
    %create data
    solution = randomdata(510,1,10);
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
Avg_Train_MSE_10 = mean(MSE_train_d_10);
Avg_Test_MSE_10 = mean(MSE_test_d_10);

%% Section 5: Visualise Results (Not part of the Question)
close all
% Visualise Results:

runx = 1:run;
plot(runx, MSE_train_d_10, 'o')
hold on
plot(runx, MSE_train_d_100, '^')
title('Training MSE of Sizes 10 & 100 Across 200 Iteration')
xlabel('Iteration')
ylabel('MSE')
legend('Training Size: 10', 'Training Size: 100' )

figure
plot(runx, MSE_test_d_10, 'o')
hold on
plot(runx, MSE_test_d_100, '^')
title('Test MSE of Sizes 10 & 100 Across 200 Iteration')
xlabel('Iteration')
ylabel('MSE')
legend('Training Size: 10', 'Training Size: 100' )



















