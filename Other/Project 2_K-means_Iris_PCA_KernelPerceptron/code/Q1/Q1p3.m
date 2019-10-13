%% Question 1 Part 1.3 Test K means on Iris dataset
 
clear all; close all;
%% Problem Setup
% load data:
load('fisheriris.mat'); 
% data already loads in desired format and a separate cell is loaded for
% species names/clustering
% load 1st 4 columns of data:
D = meas;
% convets speacies to numerical class
% from inspection, 1st 50 data points belong to the setosa group, the next 50
% to the versicolor and the last 50 to virginica. this simplifies
% assignment:
S = ones(150,1); % S is vector of true labels
S([51:100],1) = 2;
S([101:150],1) = 3;

%apply to results from part 1 and 2:
% as before: 

% setup, initialise variables:
rng(7)   %ensures below function always generates same 100 ransom seeds
RNG = randperm(1000,100); %generate 100 random time seeds between 1 and 1000 
OCCE100Ir = zeros(100,1); %final occe error output vector for 100 iterations
k = 3; % number of clusters

%% Loop and plot

for j = 1:100
   
%fit kmenas model: c is the centroid and Y(:,end)=C labels vector
[Y c] = MyKmeansAd2(D,k); 
C = Y(:,end); %predicted lables vector  


%%%%%% Plot %%%%%%  
% only plot for one data instance:
if j == 1
    figure('position', [100, 1000, 1200, 700])
    %Improve plots to account for 4D data, make 2 times 3D plots
    symbol = {'bo','rd','c^'}; 
    subplot(1,2,1)
    for i = 1:k
        temp = find(C == i);
        plot3(D(temp,1),D(temp,2),D(temp,3),symbol{i});
        hold on
    end
    plot3(c(:,1),c(:,2),c(:,3),'kx','MarkerSize',15,'LineWidth',3)
    legend('setosa', 'versicolor', 'virginica','Centroids',...
        'Location','NW')
   hold off
   xlabel('Sepal Length');
   ylabel('Sepal Width');
   zlabel('Petal Length');
   title 'Clustering of Species from Iris dataset without Petal width' 
   view(-120,30);
   grid on
    
   subplot(1,2,2)
    for i = 1:k
        temp = find(C == i);
        plot3(D(temp,1),D(temp,2),D(temp,4),symbol{i});
        hold on
    end
    plot3(c(:,1),c(:,2),c(:,4),'kx','MarkerSize',15,'LineWidth',3)
    legend('setosa', 'versicolor', 'virginica','Centroids',...
        'Location','NW')
   hold off
   xlabel('Sepal Length');
   ylabel('Sepal Width');
   zlabel('Petal Width');
   title 'Clustering of Species from Iris dataset without Petal Length' 
   view(-120,30);
   grid on

end

%% Calculate error 

%%%%%% Calculate errors %%%%%%

    OCCE100Ir(j,1)=MyOcceAd(S,C,k);
end

%answer:
aveOCCEIr = mean(OCCE100Ir);
sdOCCEIr = std(OCCE100Ir);

fprintf('The mean OCCE error on Iris data is %.3f\n',aveOCCEIr);
fprintf('The standard deviation of the OCCE error on Iris data is %.3f\n',sdOCCEIr);
