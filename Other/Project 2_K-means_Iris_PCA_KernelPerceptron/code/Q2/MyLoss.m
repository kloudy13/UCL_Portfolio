function [LossFun, minDist] = MyLoss(dataC,centr,k)
% function similar to MyDistance but now instead of returning cluster, min
% distance to cluster centroid is returned to calculate the value of the
% objective function.
% Input: 
% dataC is data matrix where rows are instances of observations
% centr is matrix of centroids (where rows correspond to no of classes and
% columns to x,y coordinates)
% k is the number of clusters
% Output: 
% Value of objective/loss function

d = zeros(k,1); % distance vector
minDist = zeros(size(dataC,1),1); % vector of min distances 
LossFun=0; % initialise loss function variable

 for i=1:size(dataC,1) %loop over all data points
     for kk=1:k %loop over all clusters
         d(kk,1)=sum((dataC(i,:)-centr(kk,:)).^2);
     end
     [val, ~] = min(d);
     minDist(i) = val;
     LossFun=LossFun+val; %sum all min distances to get objective function
end