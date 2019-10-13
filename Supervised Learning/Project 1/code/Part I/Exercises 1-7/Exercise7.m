%% Exercise 7
% We split the code into 6 sections to allow for greater readability.
% Orginally had all siz part within one for loop but found that it would be
% hard for someone else to read. Also running them separately did not
% increase the computational time by much.
%
% Section 1: "by minimizing the training error (training size 100)"
% Section 2: "by minimizing the training error (training size 10)"
% Section 3: "by minimizing the validatoin error (training size 100)"
% Section 4: "by minimizing the validatoin error (training size 10)"
% Sectoin 5: "by minimizing the 5-fold cross validation error (size 100)"
% Sectoin 6: "by minimizing the 5-fold cross validation error (size 10)"
clc

%% Section 1 - RR by minimizing the training error (training size 100)

clear
run = 200;

%initialise test MSE
MSE_test = zeros(run,1);

for iter = 1:run
    data = randomdata(600,10,100);

    % unpack results:
    Xtrain = data.Xtrain;
    ytrain = data.ytrain;
    Xtest = data.Xtest;
    ytest = data.ytest;

    lambda = 10.^(-6:3);

    Avg_MSE_train= zeros(10,1);
    MSE_train = zeros(10,1);

    for i = 1:length(lambda)
        % Calculate w for current lamda
        w_star = ridgesolution(ytrain, Xtrain,lambda(i));

        % Calculate the MSE and store the results in a matrix:
        MSE_train(i,1) = myMSE(ytrain, Xtrain, w_star);
        Avg_MSE_train(i,1) = Avg_MSE_train(i,1) + ...
            MSE_train(i,1); % Training Error   
    end

    % find minimum:
    [~, min_index] = min(MSE_train);
    low_lambda = lambda(min_index);

    % Perform RR on the test set with the optimal lambda:
    w_star = ridgesolution(ytrain, Xtrain,low_lambda);
    
    % find and store MSE
    MSE_test(iter,1) = myMSE(ytest, Xtest, w_star);
    
end

avg_test_MSE = mean(MSE_test);
std_test = std(MSE_test);

disp('Minimising Training error (100)')
fprintf('Average Test Error = %d\n', avg_test_MSE)
fprintf('Standard deviation of Test Error = %d\n', std_test)
disp('-------------------------------------------')

%% Section 2 - RR by minimizing the training error (training size 10)

clear
run = 200;

%initialise test MSE
MSE_test = zeros(run,1);

for iter = 1:run
    % ==========================================
    % The only change from the previous section:
    data = randomdata(510,10,10); 
    % ==========================================

    % unpack results:
    Xtrain = data.Xtrain;
    ytrain = data.ytrain;
    Xtest = data.Xtest;
    ytest = data.ytest;

    lambda = 10.^(-6:3);

    Avg_MSE_train= zeros(10,1);
    MSE_train = zeros(10,1);

    for i = 1:length(lambda)
        % Calculate w for current lamda
        w_star = ridgesolution(ytrain, Xtrain,lambda(i));

        % Calculate the MSE and store the results in a matrix:
        MSE_train(i,1) = myMSE(ytrain, Xtrain, w_star);
        Avg_MSE_train(i,1) = Avg_MSE_train(i,1) + ...
            MSE_train(i,1); % Training Error   
    end

    % find minimum:
    [~, min_index] = min(MSE_train);
    low_lambda = lambda(min_index);

    % Perform RR on the test set with the optimal lambda:
    w_star = ridgesolution(ytrain, Xtrain,low_lambda);
    
    % find and store MSE
    MSE_test(iter,1) = myMSE(ytest, Xtest, w_star);
    
end

Avg_MSE_train = Avg_MSE_train./run;

avg_test_MSE = mean(MSE_test);
std_test = std(MSE_test);

disp('Minimising Training error (10)')
fprintf('Average Test Error = %d\n', avg_test_MSE)
fprintf('Standard deviation of Test Error = %d\n', std_test)
disp('-------------------------------------------')

%% Section 3: RR by minimising validation error (training set 100)

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
        vali_MSE(1,i) = myMSE(y_vali, X_vali,w_star); % current vali MSE
        MSE(i,2) = MSE(i,2) + vali_MSE(1,i); % validation MSE

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
avg_test_MSE = mean(MSE_test);
std_test_MSE = std(MSE_test);

% Display Average MSE
disp('Minimising validation error (100)')
fprintf('Average Test Error = %d\n', avg_test_MSE)
fprintf('Standard deviation of Test Error = %d\n', std_test_MSE)
disp('-------------------------------------------')

%% Section 4: RR by minimising the validation error (training size 10)

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
        vali_MSE(1,i) = myMSE(y_vali, X_vali,w_star); % current vali MSE
        MSE(i,2) = MSE(i,2) + vali_MSE(1,i); % validation MSE

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
avg_test_MSE = mean(MSE_test);
std_test_MSE = std(MSE_test);

% Display Average MSE
disp('Minimising Validation error (10)')
fprintf('Average Test Error = %d\n', avg_test_MSE)
fprintf('Standard deviation of Test Error = %d\n', std_test_MSE)
disp('-------------------------------------------')

%% Section 5: RR using Cross validation (training size 100)
clear
close all

% Set lambda
lambda = 10.^(-6:3);

% number of runs
run = 200;

for iter = 1:run


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


    % initialise MSE:
    MSE_vali = zeros(length(lambda),k);

    for j = 1:k
        % Cross validation:
        % Set first '1:parition' rows as the validation set and the rest
        % as the new training set. After training on the new training set
        % shift all the rows up and place the last 'validation set' to
        % the bottom of the old training set.
        new_vali = Xtrain(1: partition,:); % new vali
        new_Xtrain = Xtrain(partition+1:NumOfData,:); % new train
        new_yvali = ytrain(1: partition); % new obs for vali
        new_ytrain = ytrain(partition+1: NumOfData); % new obs for train

        for i = 1:length(lambda)
            % Calculate w for current lamda
            w_star = ridgesolution(new_ytrain, new_Xtrain,lambda(i));

            % Calculate the MSE and store the results in a matrix:
            MSE_vali(i,j) = myMSE(new_yvali,new_vali,w_star);
        end


        % Cross validation:
        % Shift all rows up. Put the most recent "newtrain" set to the
        % bottom of the old training set.
        Xtrain(1:size(new_Xtrain,1),:) = new_Xtrain;
        Xtrain(size(new_Xtrain,1) + 1:NumOfData,:) = new_vali;
        ytrain(1:size(new_Xtrain,1),:) = new_ytrain;
        ytrain(size(new_Xtrain,1) + 1:NumOfData,:) = new_yvali;

    end

    % Average validation MSE for each lambda
    avg_MSE_vali = sum(MSE_vali,2);

    % Find which lambda gives lowest result
    [~, min_index] = min(avg_MSE_vali);
    low_lambda = lambda(min_index);

    % Perform RR on the larger training set:
    w_star = ridgesolution(ytrain, Xtrain, low_lambda);

    % find MSE on the test set
    MSE_test(iter,1) = myMSE(ytest,Xtest,w_star);
    
end

avg_test_MSE = mean(MSE_test);
std_test_MSE = std(MSE_test);

% Display Average MSE
disp('Cross validaiton (100)')
fprintf('Average Test Error = %d\n', avg_test_MSE)
fprintf('Standard deviation of Test Error = %d\n', std_test_MSE)
disp('-------------------------------------------')

%% Section 6: RR using Cross validation (training size 10)
clear
close all

% Set lambda
lambda = 10.^(-6:3);

% number of runs
run = 200;

for iter = 1:run


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


    % initialise MSE:
    MSE_vali = zeros(length(lambda),k);

    for j = 1:k
        % Cross validation:
        % Set first '1:parition' rows as the new training set and the rest
        % as the validation set. After training on the new training set
        % shift all the rows up and place the last 'new training set' to
        % the bottom of the old training set.
        new_vali = Xtrain(1: partition,:); % new train
        new_Xtrain = Xtrain(partition+1:NumOfData,:); % new vali
        new_yvali = ytrain(1: partition); % new obs for train
        new_ytrain = ytrain(partition+1: NumOfData); % new obs for vali

        for i = 1:length(lambda)
            % Calculate w for current lamda
            w_star = ridgesolution(new_ytrain, new_Xtrain,lambda(i));

            % Calculate the MSE and store the results in a matrix:
            MSE_vali(i,j) = myMSE(new_yvali,new_vali,w_star);
        end


        % Cross validation:
        % Shift all rows up. Put the most recent "newtrain" set to the
        % bottom of the old training set.
        Xtrain(1:size(new_Xtrain,1),:) = new_Xtrain;
        Xtrain(size(new_Xtrain,1) + 1:NumOfData,:) = new_vali;
        ytrain(1:size(new_Xtrain,1),:) = new_ytrain;
        ytrain(size(new_Xtrain,1) + 1:NumOfData,:) = new_yvali;

    end

    % Average validation MSE for each lambda
    avg_MSE_vali = sum(MSE_vali,2);

    % Find which lambda gives lowest result
    [~, min_index] = min(avg_MSE_vali);
    low_lambda = lambda(min_index);

    % Perform RR on the larger training set:
    w_star = ridgesolution(ytrain, Xtrain, low_lambda);

    % find MSE on the test set
    MSE_test(iter,1) = myMSE(ytest,Xtest,w_star);
    
end

avg_test_MSE = mean(MSE_test);
std_test_MSE = std(MSE_test);

% Display Average MSE
disp('Cross validaiton (10)')
fprintf('Average Test Error = %d\n', avg_test_MSE)
fprintf('Standard deviation of Test Error = %d\n', std_test_MSE)
disp('-------------------------------------------')

















