% Exercise 6

% Section 1: Training set size 100
% Section 2: Training set size 10

%% Section 1: Training Set Size 100
clear
close all


data = randomdata(600,10,100);

% unpack results:
Xtrain = data.Xtrain;
ytrain = data.ytrain;
Xtest = data.Xtest;
ytest = data.ytest;

%length of data
NumOfData = size(Xtrain,1);

% Number of partitions:
k = 5;
partition = NumOfData/k;

% Set lambda
lambda = 10.^(-6:3);

% initialise MSE:
MSE_matrix = zeros(10,3);

for j = 1:k
    % Cross validation:
    % Set first '1:parition' rows as the new training set and the rest as
    % the validation set. After training on the new training set shift all
    % the rows up and place the last 'new training set' to the bottom of
    % the old training set.
    new_Xtrain = Xtrain(1: partition,:); % new train
    new_vali = Xtrain(partition+1:NumOfData,:); % new vali
    new_ytrain = ytrain(1: partition); % new obs for train
    new_yvali = ytrain(partition+1: NumOfData); % new obs for vali

    for i = 1:length(lambda)
        % Calculate w for current lamda
        w_star = ridgesolution(new_ytrain, new_Xtrain,lambda(i));

        % Calculate the MSE and store the results in a matrix:
        MSE_matrix(i,1) = MSE_matrix(i,1) + ...
            myMSE(new_ytrain, new_Xtrain, w_star); % trainging error
        MSE_matrix(i,2) = MSE_matrix(i,2) + ...
            myMSE(new_yvali, new_vali, w_star);  % validation error
        MSE_matrix(i,3) = MSE_matrix(i,3) +...
            myMSE(ytest, Xtest, w_star); % Test Error


    end
     
    
    % Cross validation:
    % Shift all rows up. Put the most recent "newtrain" set to the bottom
    % of the old training set.
    Xtrain(1:size(new_vali,1),:) = new_vali;
    Xtrain(size(new_vali,1) + 1:NumOfData,:) = new_Xtrain;
    ytrain(1:size(new_vali,1),:) = new_yvali;
    ytrain(size(new_vali,1) + 1:NumOfData,:) = new_ytrain;
    
end

% Find average MSE
MSE_matrix = MSE_matrix./k;

% THE QUESTION HAS NOT ASKED TO PERFORM RR WITH THE OPTIMAL LAMBDA.

% Plot results:
% Plot training and test as function of lambda
plot(log(lambda), MSE_matrix(:,1))
hold on
plot(log(lambda), MSE_matrix(:,2))
hold on
plot(log(lambda), MSE_matrix(:,3))
title('Cross validation with Training Set Size 100') 
xlabel('log(lambda)')
ylabel('MSE')
legend('Training MSE','Validation MSE', 'Test MSE','Location', 'southeast')

% Display
ax = gca;
ax.XAxisLocation = 'origin';
grid on
hold off

%% Section 2: Training Set Size 10

clear
close all


data = randomdata(510,10,10);

% unpack results:
Xtrain = data.Xtrain;
ytrain = data.ytrain;
Xtest = data.Xtest;
ytest = data.ytest;

%length of data
NumOfData = size(Xtrain,1);

% Number of partitions:
k = 5;
partition = NumOfData/k;

% Set lambda
lambda = 10.^(-6:3);

% initialise MSE:
MSE_matrix = zeros(10,3);

for j = 1:k
    % Cross validation:
    % Set first '1:parition' rows as the new training set and the rest as
    % the validation set. After training on the new training set shift all
    % the rows up and place the last 'new training set' to the bottom of
    % the old training set.
    new_Xtrain = Xtrain(1: partition,:); % new train
    new_vali = Xtrain(partition+1:NumOfData,:); % new vali
    new_ytrain = ytrain(1: partition); % new y for train
    new_yvali = ytrain(partition+1: NumOfData); % new y for vali

    for i = 1:length(lambda)
        % Calculate w for current lamda
        w_star = ridgesolution(new_ytrain, new_Xtrain,lambda(i));

        % Calculate the MSE and store the results in a matrix:
        MSE_matrix(i,1) = MSE_matrix(i,1) + ...
            myMSE(new_ytrain, new_Xtrain, w_star); % trainging error
        MSE_matrix(i,2) = MSE_matrix(i,2) + ...
            myMSE(new_yvali, new_vali, w_star);  % validation error
        MSE_matrix(i,3) = MSE_matrix(i,3) +...
            myMSE(ytest, Xtest, w_star); % Test Error


    end
     
    
    % Cross validation:
    % Shift all rows up. Put the most recent "newtrain" set to the bottom
    % of the old training set.
    Xtrain(1:size(new_vali,1),:) = new_vali;
    Xtrain(size(new_vali,1) + 1:NumOfData,:) = new_Xtrain;
    ytrain(1:size(new_vali,1),:) = new_yvali;
    ytrain(size(new_vali,1) + 1:NumOfData,:) = new_ytrain;
    
end

% Find average MSE
MSE_matrix = MSE_matrix./k;

% THE QUESTION HAS NOT ASKED TO PERFORM RR WITH THE OPTIMAL LAMBDA.

% Plot results:
% Plot training and test as function of lambda
plot(log(lambda), MSE_matrix(:,1))
hold on
plot(log(lambda), MSE_matrix(:,2))
hold on
plot(log(lambda), MSE_matrix(:,3))
title('Cross validation with Training Set Size 10') 
xlabel('log(lambda)')
ylabel('MSE')
legend('Training MSE','Validation MSE', 'Test MSE','Location', 'southeast')

% Display
ax = gca;
ax.XAxisLocation = 'origin';
grid on
hold off

