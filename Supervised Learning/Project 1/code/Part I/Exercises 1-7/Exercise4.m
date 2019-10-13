%% Exercise 4

%% part a

clear

solution = randomdata(600,10,100);

% unpack results:
Xtrain = solution.Xtrain;
ytrain = solution.ytrain;
Xtest = solution.Xtest;
ytest = solution.ytest;

lambda = 10.^(-6:3);

MSE_ridge_a = zeros(10,2);

for i = 1:length(lambda)
    % Calculate w for current lamda
    w_star = ridgesolution(ytrain, Xtrain,lambda(i));
    
    % Calculate the MSE and store the results in a matrix:
    MSE_ridge_a(i,1) = myMSE(ytrain, Xtrain, w_star); % Training Error
    MSE_ridge_a(i,2) = myMSE(ytest, Xtest, w_star); % Test Error
    
    
end

% Plot training and test as function of lambda
plot(log(lambda), MSE_ridge_a(:,1))
hold on
plot(log(lambda), MSE_ridge_a(:,2))
title('Training and Test error with Training Set Size 100') 
xlabel('log(lambda)')
ylabel('MSE')
legend('Training MSE', 'Test MSE','Location', 'southeast')

% Display
ax = gca;
ax.XAxisLocation = 'origin';
grid on
hold off

%% PART B
clear
close all

% copy paste from above. Main change is the training size of 10.

solution = randomdata(510,10,10);

% unpack results:
Xtrain = solution.Xtrain;
ytrain = solution.ytrain;
Xtest = solution.Xtest;
ytest = solution.ytest;

lambda = 10.^(-6:3);

MSE_ridge_b = zeros(10,2);

for i = 1:length(lambda)
    % Calculate w for current lamda
    w_star = ridgesolution(ytrain, Xtrain,lambda(i));
    
    % Calculte MSD and store results in 
    MSE_ridge_b(i,1) = myMSE(ytrain, Xtrain, w_star); % Training Error
    MSE_ridge_b(i,2) = myMSE(ytest, Xtest, w_star); % Test Error
end

% Plot data 
figure
plot(log(lambda), log(MSE_ridge_b(:,1)))
hold on
plot(log(lambda),log(MSE_ridge_b(:,2)))
title('Training and Test error with Training tet Size 10 (log scales)')
xlabel('log(lambda)')
ylabel('log(MSE)')
legend('Training MSE', 'Test MSE','Location', 'southeast')

%Display axis
ax = gca;
ax.XAxisLocation = 'origin';
grid on
hold off

% plot data without log MSE for comparision to part (a)
figure
plot(log(lambda), MSE_ridge_b(:,1))
hold on
plot(log(lambda),MSE_ridge_b(:,2))
title('Training and Test error with Training tet Size 10')
xlabel('log(lambda)')
ylabel('MSE')
legend('Training MSE', 'Test MSE','Location', 'southeast')

%Display axis
ax = gca;
ax.XAxisLocation = 'origin';
grid on
hold off






%% Part c
clear
close all

run = 200;

% For training set size of 100
%initialise MSE 

MSE_ridge_c_100 = zeros(10,2);

lambda = 10.^(-6:3);

for j = 1:run
    
    solution = randomdata(600,10,100);

    % unpack results:
    Xtrain = solution.Xtrain;
    ytrain = solution.ytrain;
    Xtest = solution.Xtest;
    ytest = solution.ytest;

    for i = 1:length(lambda)
        % Calculate w for current lamda
        w_star = ridgesolution(ytrain, Xtrain,lambda(i));

        %Calculate total MSE
        MSE_ridge_c_100(i,1) = MSE_ridge_c_100(i,1) + myMSE(ytrain, Xtrain, w_star);
        MSE_ridge_c_100(i,2) = MSE_ridge_c_100(i,2) + myMSE(ytest, Xtest, w_star);

    end
end

% Find average
MSE_ridge_c_100 = MSE_ridge_c_100/run;

% Calculate MSE for training set size of 10. The code is a copy-paste of
% the code above, just needed to changed input for randomdata.m and
% variable names. Keep code separate as its easier to manage and only
% marginally affects performance.

MSE_ridge_c_10 = zeros(10,2);

for j = 1:run
    
    solution = randomdata(510,10,10);

    % unpack results:
    Xtrain = solution.Xtrain;
    ytrain = solution.ytrain;
    Xtest = solution.Xtest;
    ytest = solution.ytest;

    for i = 1:10
        % Calculate w for current lamda
        w_star = ridgesolution(ytrain, Xtrain,lambda(i));
        
        %Calculate total MSE
        MSE_ridge_c_10(i,1) = MSE_ridge_c_10(i,1) + ...
            myMSE(ytrain, Xtrain, w_star);
        MSE_ridge_c_10(i,2) = MSE_ridge_c_10(i,2) + ...
            myMSE(ytest, Xtest, w_star);

    end
end

% Find average of MSE
MSE_ridge_c_10 = MSE_ridge_c_10/run;

% Plot graphs for above data on separate figures;

% Plot data training set size of 100
plot(log(lambda), log(MSE_ridge_c_100(:,1)));
hold on
plot(log(lambda), log(MSE_ridge_c_100(:,2)));
title('Train and Test error (Training Size: 100, 200 runs)')
xlabel('log(lambda)')
ylabel('log(MSE)')
legend('Training MSE', 'Test MSE','Location', 'northwest')

%Display axis:
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
grid on

% Plot data training set size of 10
figure
plot(log(lambda), log(MSE_ridge_c_10(:,1)));
hold on
plot(log(lambda), log(MSE_ridge_c_10(:,2)));
title('Train and Test error (Training Size: 10, 200 runs)')
xlabel('log(lambda)')
ylabel('log(MSE)')
legend('Training MSE', 'Test MSE','Location', 'southeast')

%Display axis:
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
grid on



    
    










