function [posi] = MyDist2(X,c)
% Purpose: calculates distance between points in X and c, finds minimum and 
% assigns the corresponding class label, retuning the clustering vector.
% Input: 
% X is data matrix where rows are instances of observations
% c is matrix of centroids (where rows correspond to no of classes and
% columns to x,y coordinates)
% Output: 
% Vector of cluster class 


%setup:
d = zeros(size(c,1),1); %vector of distance to each of k clusters 
posi = zeros(size(X,1),1);  %vector to corresponding class assignment

    for i=1:size(X,1) %loop over all data points
        for k=1:size(c,1) %loop over all clusters
            d(k,1)=sum((X(i,:)-c(k,:)).^2);
        end
        [~,pos]=min(d);
        posi(i,1)=pos;
    end

end

