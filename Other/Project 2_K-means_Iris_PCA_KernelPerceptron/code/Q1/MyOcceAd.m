function [occe]=MyOcceAd(S,C,k)
% Function calculates the occe error for kmeans clustering algorithm
% it initalises a matrix of all possible cluster permitations based on k
% Input:
% S - True classification lables
% k - no of clusters
% Output: occe error 

p=perms(1:k);%create matrix of possible lable permutations (6 by 3)

err=zeros(size(p,1),1);
temp=C;
len=size(C,1);

for i=1:size(p,1) 
    temp(C==1)=p(i,1);
    temp(C==2)=p(i,2);
    temp(C==3)=p(i,3);
    err(i,1)=(sum(temp~=S))/len;
end
occe=min(err);

    