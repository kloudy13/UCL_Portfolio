clear all;
close all;
%% 1-NN Algorithm Sample Complexity

% Setup & initialise variables:
rng(1); % set time seed
n = 14; % largest number of column dimensions of data set
m = 100; % number of test data instances used to find generalisation error
loop = 100; % iterations per each n
GenErrLim = 0.1; % point where 10% error reached as per handout defintion 
NumSamples = zeros(n, loop); % initialise Sample Complexity variable

for i = 1:n   % change size of columns of data set X

    for l = 1:loop

        % generate test and training data  from a uniformly randomly
        % genrated data set ([-1,1]^n)
        startSize = m;  
        
        trainX = randi([0 1], startSize, i); %uniform random set of [0,1]
        testX = randi([0 1], m, i);
       
        trainX(find(trainX == 0)) = -1; % convert to set of [-1,1]
        testX(find(testX == 0)) = -1;
        
        trainY = trainX(:, 1); %select first column
        testY = testX(:, 1);
        
        % initalise error and counter vlaues: 
        Counter = 0;  % sample complexity counter
        GenErr = 1000000000; % artificaly high number 
        loss= 0;
       
       
        while(GenErr > GenErrLim)
 
           % Keep track of the number of training samples used in the loop: 
           Counter = Counter + 1;
           
           % Expand training data if more are samples needed than avaialble
           if Counter > startSize
               
               addSize = 500; % increase current size by mNew rows
               
               % generate more data:
               newX = randi([0 1], addSize, i); %uniform random set of [0,1]
               newX(find(newX == 0)) = -1; % convert to set of [-1,1]
               newY = newX(:, 1); %select first column
      
               trainX = [trainX; newX]; %update dataset 
               trainY = [trainY; newY];
               
               startSize = startSize + addSize; % update size of training data
               
           end
            
           loss = nn1(trainX(1:Counter,:), trainY(1:Counter), testX, testY);
                
            % calulate the generalisation error estimate as a proportion 
            % of the amount of test data instances (m).
            GenErr = loss /m;        
        end
        
        % sample complexity as a function of the n dimension of data:
        NumSamples(i,l) = Counter;
        
    end
    
end

% calculate means and standard deviations of number of samples needed
 meanNumSamples = mean(NumSamples,2);


figure;
plot(meanNumSamples)
ylabel('m');
xlabel('n');
title({'Number of samples (m) to obtain 10% generalisation error versus data dimension (n)','1-NN algorithm'});


