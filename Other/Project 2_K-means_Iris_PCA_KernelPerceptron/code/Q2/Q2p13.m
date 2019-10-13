%% Question 2 Part 1.3: Implement PCA and K means on Iris data
clear all; close all;
%% Setup 
% Setup for Iris dataset:
% load data:
load('fisheriris.mat'); 
% data already loads in desired format and a separate cell is loaded for
% species names/clustering
% load 1st 4 columns of data:
D = meas;
% converts species strings to numerical class
% from inspection, 1st 50 data points belong to the setosa group, the next 50
% to the versicolor and the last 50 to virginica. this simplifies
% assignment:
S = ones(150,1); % S is vector of true labels
S([51:100],1) = 2;
S([101:150],1) = 3;

% Further problem setup:

% K-means setup:
k = 3; % no of clusters/groups

% PCA setup:
kp = [0:4]; % vector of PCA parameters (dim. reduction) 
            % case when k=0 is when perform analysis on raw Iris data
            % without PCA conversion.

% center dataset:
dataC = D-mean(D);

% initiate occe and loss parameters
occe = zeros(100,length(kp));
EX = zeros(100,length(kp));

%% Part 2.1.3-loop 100 times:
% Loop to find occe and objective function:
for j=1:100
    %loop over the desired PC parameters (kp) calculating occe and objective
    %fun. for each iteration:
    for i=1:length(kp)

        if i==1 % case when kp=0, 
            tempData = D; % set input argument for K-menas function to raw data
                          % function MyPCA below does not give error when 
                          % kp=0 so this should work well.
        else
            % perform PCA with given kp parameter:
            tempData = MyPCA(dataC,(i-1));  
        end

        % perform Kmeans for PCA or normal data with given k parameter:
        [y,centr] = MyKmeansAd2(tempData,k);
        C = y(:,end); % last column of y contains the classification labels

        % calculate occe error:
        occe(j,i) = MyOcceAd(S,C,k); 
        EX(j,i) = MyLoss(tempData,centr,k);
    end
end
%% Part 2.1.3(a) Find smallest 3 occe values and corresponding objective fun.
%
% there are only 3 distinct values of the occe vector as the data has 3
% classes (k-menas k=3). Furthermore, this result is not surprising as we 
% are looping over the same dataset and little variability is introduced. 
% In asking for the 3 smallest occe values the question is ambiguous. 
% I assume, it is  asking for the 3 minimum non equivalent occe values 
% (not vectors), in which case these can be found in the vector below. 
% If it is simply asking for the 3 lowest repeated values this will be 3 
% instances of the first row of output below.

%Min3occe = svds(occe,3,'smallest');
Min3occe = sort(unique(min(occe)));
Min3occe = Min3occe(1:3)';

%corresponding corresponding objective function:
Min3Ex = zeros(3,1);
    for ii=1:3
        index = find(Min3occe(ii,1) == occe,1);
        Min3Ex(ii,1) = EX(index);
        
        clearvars index
    end   
% store as matrix:    
Min3vals=[ Min3occe,  Min3Ex];

% output as table:   
Row = {'Min1';'Min2';'Min3'};
Occe = Min3occe;
Objective = Min3Ex;

disp('Three lowest occe values anc corresponding objective function:')
T = table(Occe,Objective,'RowNames',Row)

%% Part 2.1.3(b) Find corresponding mean and standard deviation.
meanMin3occe = mean(Min3occe);
sdMin3occe = std(Min3occe);

meanMin3Ex = mean(Min3Ex);
sdMin3Ex = std(Min3Ex);

fprintf('The mean and standard deviation of the smallest 3 occe are: %.3f and %.3f respectively, \nwhereas the corresponding mean and standard deviation of the objective function are: %.3f and %.3f \n',meanMin3occe,sdMin3Ex,meanMin3Ex,sdMin3Ex);

%% Part 2.1.3(c) Make chart of occe as a rank function.

SortOcce = zeros(size(occe));
SortEx = zeros(size(EX));

% below loop sorts and ranks occe and objective fun. as per question
% requirements:
for jj = 1:length(kp)
 
    % select corresponding columns of occe and objective function: 
    tempEx = EX(:,jj);
    tempOcce = occe(:,jj);
    % concatenate:
    temp = [tempEx, tempOcce];
    % sort in ascending order:
    in = sortrows(temp);
    % store occe result:   
    SortOcce(:,jj) = in(:,2);
    % converts objective fun. values to a rank and store result:
    uniqueEx = unique(in(:,1)); % vector of distinct loss function values
    for p=1:length(uniqueEx)
        index = find(in(:,1) == uniqueEx(p,1));
        SortEx(index,jj) = p; % overwrite loss fun.values with rank index
    end
    clearvars temp tempEx tempOcce in uniqueEx 
end

% produce desired plot:
figure('position', [50, 500, 1000, 600]);
for pp = 1:length(kp)
    hold on
    subplot(2,3,pp)
    bar(SortOcce(:, pp));
    xlabel('Objective Function rank'); 
    ylabel('occe');
    title(sprintf('Occe vs Objective function rank %pp\n',pp));
end
% it should be noted that although the objective function has been ranked
% as per question requirements, the bar plot command only admits the occe
% values and counts these (thus, we have a count of up to 150 on the x axis)
% assigning the rank therefore seams to have been redundant.  

%% Part 2.1.4 Visualise data for reduced dimensions
% in particular, when dimensions reduced to kp = 2 ,3.
 
% regenerate data:
D2 = MyPCA(dataC,2);  
[Y2,centr2] = MyKmeansAd2(D2,k);
D3 = MyPCA(dataC,3);  
[Y3,centr3] = MyKmeansAd2(D3,k);

% plot for 2D, k=2:
figure('position', [50, 500, 600, 500]);
hold on
plot(D2(Y2(:,end)==1,1),D2(Y2(:,end)==1,2),'ro','MarkerSize',9)
plot(D2(Y2(:,end)==2,1),D2(Y2(:,end)==2,2),'go','MarkerSize',9)
plot(D2(Y2(:,end)==3,1),D2(Y2(:,end)==3,2),'bo','MarkerSize',9)
plot(centr2(:,1),centr2(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3)
legend('Cluster 1','Cluster 2','Centroids',...
       'Location','NW')
title 'Clustering of Species from PCA on Iris dataset with k=2'
xlabel('PCA(x1)')
ylabel('PCA(x2)')
hold off

% plot for 3D, k=3:

symbol = {'bo','rd','c^'}; 

figure('position', [50, 500, 600, 500])
    for i = 1:3
        temp = find(Y3(:,end) == i);
        plot3(D3(temp,1),D3(temp,2),D3(temp,3),symbol{i});
        hold on
    end
    plot3(centr3(:,1),centr3(:,2),centr3(:,3),'kx','MarkerSize',15,'LineWidth',3)
    legend('Cluster 1','Cluster 2', 'Cluster 3','Centroids',...
        'Location','NW')
   hold off
   xlabel('PCA(x1)');
   ylabel('PCA(x2)');
   zlabel('PCA(x3)');
   title 'Clustering of Species from PCA on Iris dataset with k=3' 
   view(-120,30);
   grid on