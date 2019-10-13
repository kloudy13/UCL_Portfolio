%% Exercise 5

% 

%% Question 5 - part a 
% Ridge Regression on 10 dimensional data with validation set

clear
close all

% Number of iterations:
iter = 200;

% Create array of lambda
lambda = 10.^(-6:3);

% Initialise vector for storing average MSE for each lamda
MSE = zeros(length(lambda),3);

% Initialise array for storing validation MSE for each iteration
vali_MSE = zeros(1,length(lambda));

% Initialise vector for storing lowest lambda and test MSE
low_lambda = zeros(iter,1);
MSE_test = zeros(iter,1);

for j = 1:iter
    
    data = randomdata(600,10,100);

    % unpack results:
    Xtrain = data.Xtrain;
    ytrain = data.ytrain;
    Xtest = data.Xtest;
    ytest = data.ytest;

    % Split data into training and test using 'datasplit.m'

    %X_train_V is the new smaller training set:
    [X_train_V, X_vali] = datasplit(Xtrain, 80); % split training data
    [y_train_V, y_vali] = datasplit(ytrain, 80); % split observations
    
        % Find MSE for each lambda
     for i = 1:length(lambda)

        %Find optimatal w* on the new smaller training set:
        w_star = ridgesolution(y_train_V, X_train_V,lambda(i));

        %Find and store MSE for each lamda:
        %training MSE:
        MSE(i,1) = MSE(i,1) + myMSE(y_train_V, X_train_V,w_star); 
        vali_MSE(1,i) = myMSE(y_vali, X_vali,w_star); % current vali MSE
        MSE(i,2) = MSE(i,2) + vali_MSE(1,i); % validation MSE
        MSE(i,3) = MSE(i,3) + myMSE(ytest, Xtest, w_star); % Test MSE

    end

    % Find index of lowest validation MSE
    [~ , lowestMSE] = min(vali_MSE);

    % Find and store the lowest lambda
    low_lambda(j,1) = lambda(lowestMSE);

    % Find w* on larger training set:
    w_star = ridgesolution(ytrain, Xtrain,low_lambda(j,1));

    % Calculate and store new lowest test MSE
    MSE_test(j,1) = myMSE(ytest, Xtest, w_star);
end

% Find average MSE of RR on the smaller training set
MSE = MSE./iter;

% Average test lambda & MSE 
avg_lambda = mean(low_lambda);
avg_test_MSE = mean(MSE_test);

% Plot training, validation and test MSE as function of log(lambda)
plot(log(lambda),MSE(:,1)); % training MSE
hold on
plot(log(lambda),MSE(:,2)); % Validation
hold on
plot(log(lambda),MSE(:,3)); % Test

% Make graph pretty
title('Train, Validation and Test error (Training Size: 100, 200 runs')
xlabel('log(lambda)')
ylabel('MSE')
legend('Training MSE','Validation MSE', 'Test MSE',...
        'Location', 'northwest')
ax = gca;
ax.XAxisLocation = 'origin';
grid on

% Display Average MSE
fprintf('Average Test Error = %d\n', avg_test_MSE)
fprintf('Average Lambda = %d\n', avg_lambda)






%% Question 5 -  part b - 8 training 2 validation
clear
close all

% For the most part this code is a copy of the code above. I have only
% changed the inputs for the relevant functions.

% Number of iterations:
iter = 200;

% Create array of lambda
lambda = 10.^(-6:3);

% Initialise vector for storing average MSE for each lamda
MSE = zeros(length(lambda),3);

% Initialise array for storing validation MSE for each iteration
vali_MSE = zeros(1,length(lambda));

% Initialise vector for storing lowest lambda and test MSE
low_lambda = zeros(iter,1);
MSE_test = zeros(iter,1);

for j = 1:iter
    
    data = randomdata(510,10,10);

    % unpack results:
    Xtrain = data.Xtrain;
    ytrain = data.ytrain;
    Xtest = data.Xtest;
    ytest = data.ytest;

    % Split data into training and test using 'datasplit.m'

    %X_train_V is the new smaller training set:
    [X_train_V, X_vali] = datasplit(Xtrain, 8); % split training data
    [y_train_V, y_vali] = datasplit(ytrain, 8); % split observations
    
    % Find MSE for each lambda
    for i = 1:length(lambda)

        %Find optimatal w* on the new smaller training set:
        w_star = ridgesolution(y_train_V, X_train_V,lambda(i));

        %Find and store MSE for each lamda:
        %training MSE:
        MSE(i,1) = MSE(i,1) + myMSE(y_train_V, X_train_V,w_star);
        vali_MSE(1,i) = myMSE(y_vali, X_vali,w_star); % current vali MSE
        MSE(i,2) = MSE(i,2) + vali_MSE(1,i); % validation MSE
        MSE(i,3) = MSE(i,3) + myMSE(ytest, Xtest, w_star); % Test MSE

    end

    % Find index of lowest validation MSE
    [~ , lowestMSE] = min(vali_MSE);

    % Find and store the lowest lambda
    low_lambda(j,1) = lambda(lowestMSE);

    % Find w* on larger training set:
    w_star = ridgesolution(ytrain, Xtrain,low_lambda(j,1));

    % Calculate and store new lowest test MSE
    MSE_test(j,1) = myMSE(ytest, Xtest, w_star);
end

% Find average MSE of RR on the smaller training set
MSE = MSE./iter;

% Average test lambda & MSE 
avg_lambda = mean(low_lambda);
avg_test_MSE = mean(MSE_test);

plot(log(lambda),MSE(:,1)); % training MSE
hold on
plot(log(lambda),MSE(:,2)); % Validation
hold on
plot(log(lambda),MSE(:,3)); % Test
hold on

% Make graph pretty
title('Train, Validation and Test error (Training Size: 100, 200 runs')
xlabel('log(lambda)')
ylabel('MSE')
legend('Training MSE','Validation MSE', 'Test MSE',...
    'Location', 'southeast')
ax = gca;
ax.XAxisLocation = 'origin';
grid on

% Display Average MSE
fprintf('Average Test Error = %d\n', avg_test_MSE)
fprintf('Average Lambda = %d\n', avg_lambda)

hold off

%% Repeat part (a) with a 1 dimentional data set
clear
close all

% Number of iterations:
iter = 200;

% Create array of lambda
lambda = 10.^(-6:3);

% Initialise vector for storing average MSE for each lamda
MSE = zeros(length(lambda),3);

% Initialise array for storing validation MSE for each iteration
vali_MSE = zeros(1,length(lambda));

% Initialise vector for storing lowest lambda and test MSE
low_lambda = zeros(iter,1);
MSE_test = zeros(iter,1);

for j = 1:iter
    
    data = randomdata(600,1,100);

    % unpack results:
    Xtrain = data.Xtrain;
    ytrain = data.ytrain;
    Xtest = data.Xtest;
    ytest = data.ytest;

    % Split data into training and test using 'datasplit.m'

    %X_train_V is the new smaller training set:
    [X_train_V, X_vali] = datasplit(Xtrain, 80);
    [y_train_V, y_vali] = datasplit(ytrain, 80);
    
    % Find MSE for each lambda
    for i = 1:length(lambda)

    %Find optimatal w* on the new smaller training set:
    w_star = ridgesolution(y_train_V, X_train_V,lambda(i));

    %Find and store MSE for each lamda:
    MSE(i,1) = MSE(i,1) + myMSE(y_train_V, X_train_V,w_star); %training MSE
    vali_MSE(1,i) = myMSE(y_vali, X_vali,w_star); % store MSE for current i
    MSE(i,2) = MSE(i,2) + vali_MSE(1,i); % validation MSE
    MSE(i,3) = MSE(i,3) + myMSE(ytest, Xtest, w_star); % Test MSE

    end

    % Find index of lowest validation MSE
    [~ , lowestMSE] = min(vali_MSE);

    % Find and store the lowest lambda
    low_lambda(j,1) = lambda(lowestMSE);

    % Find w* on larger training set:
    w_star = ridgesolution(ytrain, Xtrain,low_lambda(j,1));

    % Calculate and store new lowest test MSE
    MSE_test(j,1) = myMSE(ytest, Xtest, w_star);
end

% Find average MSE of RR on the smaller training set
MSE = MSE./iter;

% Average test lambda & MSE 
avg_lambda = mean(low_lambda);
avg_test_MSE = mean(MSE_test);

% Plot training, validation and test MSE as function of log(lambda)
plot(log(lambda),MSE(:,1)); % training MSE
hold on
plot(log(lambda),MSE(:,2)); % Validation
hold on
plot(log(lambda),MSE(:,3)); % Test
hold on


% Make graph pretty
title('Train, Validation and Test error (Training Size: 100, 200 runs')
xlabel('log(lambda)')
ylabel('MSE')
legend('Training MSE','Validation MSE', 'Test MSE',...
    'Location', 'northwest')
ax = gca;
ax.XAxisLocation = 'origin';
grid on

% Display Average MSE
fprintf('Average Test Error = %d\n', avg_test_MSE)
fprintf('Average Lambda = %d\n', avg_lambda)

%% Repeat part (b) with 1 dimentional data set
clear
close all

% For the most part this code is a copy of the code above. I have only
% changed the inputs for the relevant functions.

% Number of iterations:
iter = 200;

% Create array of lambda
lambda = 10.^(-6:3);

% Initialise vector for storing average MSE for each lamda
MSE = zeros(length(lambda),3);

% Initialise array for storing validation MSE for each iteration
vali_MSE = zeros(1,length(lambda));

% Initialise vector for storing lowest lambda and test MSE
low_lambda = zeros(iter,1);
MSE_test = zeros(iter,1);

for j = 1:iter
    
    data = randomdata(510,1,10);

    % unpack results:
    Xtrain = data.Xtrain;
    ytrain = data.ytrain;
    Xtest = data.Xtest;
    ytest = data.ytest;

    % Split data into training and test using 'datasplit.m'

    %X_train_V is the new smaller training set:
    [X_train_V, X_vali] = datasplit(Xtrain, 8);
    [y_train_V, y_vali] = datasplit(ytrain, 8);
    
    % Find MSE for each lambda
    for i = 1:length(lambda)

    %Find optimatal w* on the new smaller training set:
    w_star = ridgesolution(y_train_V, X_train_V,lambda(i));

    %Find and store MSE for each lamda:
    MSE(i,1) = MSE(i,1) + myMSE(y_train_V, X_train_V,w_star); %training MSE
    vali_MSE(1,i) = myMSE(y_vali, X_vali,w_star); % store MSE for current i
    MSE(i,2) = MSE(i,2) + vali_MSE(1,i); % validation MSE
    MSE(i,3) = MSE(i,3) + myMSE(ytest, Xtest, w_star); % Test MSE

    end

    % Find index of lowest validation MSE
    [~ , lowestMSE] = min(vali_MSE);

    % Find and store the lowest lambda
    low_lambda(j,1) = lambda(lowestMSE);

    % Find w* on larger training set:
    w_star = ridgesolution(ytrain, Xtrain,low_lambda(j,1));

    % Calculate and store new lowest test MSE
    MSE_test(j,1) = myMSE(ytest, Xtest, w_star);
end

% Find average MSE of RR on the smaller training set
MSE = MSE./iter;

% Average test lambda & MSE 
avg_lambda = mean(low_lambda);
avg_test_MSE = mean(MSE_test);

% Plot training, validation and test MSE as function of log(lambda)
plot(log(lambda),MSE(:,1)); % training MSE
hold on
plot(log(lambda),MSE(:,2)); % Validation
hold on
plot(log(lambda),MSE(:,3)); % Test
hold on
hold off


% Make graph pretty
title('Train, Validation and Test error (Training Size: 100, 200 runs')
xlabel('log(lambda)')
ylabel('MSE')
legend('Training MSE','Validation MSE', 'Test MSE',...
    'Location', 'northwest')
ax = gca;
ax.XAxisLocation = 'origin';
grid on

% Display Average MSE
fprintf('Average Test Error = %d\n', avg_test_MSE)
fprintf('Average Lambda = %d\n', avg_lambda)









