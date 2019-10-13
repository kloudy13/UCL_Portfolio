clear all;
close all;
%% Winnow Algorithm Sample Complexity

% Setup & initialise variables:
n = 100; % largest number of dimensions. 
m = 1000; % number of test data point used for each error estimation.  
loop = 20; % number of runs per dimension. 
GenErrLim = 0.1; % point where 10% error reached as per handout definition 
NumSamples = zeros(n, loop); % initialise Sample Complexity variable

for i = 1:n   % change size of column of data set

    for l = 1:loop

        % generate test and training data  from a uniformly randomly
        % generated data set ([-1,1]^n)  
        
        trainX = randi([0 1], m, i); %uniform random set of [0,1]
        testX = randi([0 1], m, i);
        
        trainY = trainX(:, 1); %select first column
        testY = testX(:, 1);
        
        Counter = 0; % sample complexity counter
        GenErr = 1000; % start by setting at an artificially high number 
        
        
        while(GenErr > GenErrLim)
            % Keep track of the number of training samples used in the loop: 
            Counter = Counter + 1;
            
            % train winnow (find weight vector)
            w = winnowtrain(trainX(1:Counter,:), trainY(1:Counter, :));
            
            % test winnow (apply leaned weight to test data)
            yP = ((testX * w) -i >= 0); % if able miss-match set to 0

            % errors (count miss-classification instances):
            loss = sum(yP ~= testY);
            GenErr = (loss)/m/2;
             % calculate the generalisation error estimate as a proportion 
             % of the amount of test data instances (m).    
        end
        
       % sample complexity as a function of the n dimension of data:
       NumSamples(i,l) = Counter; 
    end
end

% calculate means and standard deviations of number of samples needed
meanNumSamples = mean(NumSamples,2);
sdNumSamples = std(NumSamples,[],2);

figure;
hold on;
plot(meanNumSamples)
title({'Number of samples (m) to obtain 10% generalisation error versus data dimension (n)','Winnow Algorithm'});
ylabel('m');
xlabel('n');
errorbar(1:n, meanNumSamples, sdNumSamples);
hold off;
