
%% Question 1 Part 1.2 Test K means on specified dataset 
clear all; close all;
%% Setup, initialise variables:
rng(7)   %ensures below function always generates same 100 ransom seeds
RNG=randperm(1000,100); %generate 100 random time seeds between 1 and 1000 
OCCE100=zeros(100,1); %final occe error output vector for 100 iterations

%% Loop and plot:
for j=1:100
   
[data] = MYgenData2(RNG(1,j)); %ensures same 100 data sets always generated

k=3;
% note question does not explicitly state how many clusters we are after,
% I will assume 3 is this corresponds to the 3 data generating process

% generate augmented data matrix inc 
D = zeros(size(data,1),k);
D(:,[1:2]) = data;
D([1:50],k) = 1;
D([51:100],k) = 2;
D([101:150],k) = 3;
S=D(:,k); %true labels vector

%fit kmenas model: c is the centroid and Y(:,end)=C labels vector
[Y c] = MyKmeansAd2(D(:,[1:2]),k); 
C=Y(:,end); %predicted labels vector  

%%%%%% Plot %%%%%%  
% only plot for one data instance:
if j==1  
    figure('position', [50, 50, 900, 550])
    hold on
    plot(D(Y(:,end)==1,1),D(Y(:,end)==1,2),'ro','MarkerSize',9)
    plot(D(Y(:,end)==2,1),D(Y(:,end)==2,2),'go','MarkerSize',9)
    plot(D(Y(:,end)==3,1),D(Y(:,end)==3,2),'bo','MarkerSize',9)
    plot(c(:,1),c(:,2),'kx',...
         'MarkerSize',15,'LineWidth',3)
    legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
           'Location','NW')
    title 'Cluster Assignments and Centroids'
    xlabel('x')
    ylabel('y')
    hold off
end
%% Calculate errors
    OCCE100(j,1)=MyOcceAd(S,C,k);
end

%answer:
aveOCCE=mean(OCCE100);
sdOCCE=std(OCCE100);

fprintf(' The mean OCCE error is %.3f\n',aveOCCE);
fprintf(' The standard deviation of the OCCE error is %.3f\n',sdOCCE);