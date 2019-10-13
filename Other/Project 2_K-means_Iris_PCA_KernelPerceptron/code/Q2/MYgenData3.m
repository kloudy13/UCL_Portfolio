function [ data ] = MYgenData3(RNG,M,N,k)
% generate random numbers
% Input:
% RNG
% M is row dimension of data
% N is column dimenstoon of data MUST BE EVEN
% k is number of clusters ie. distinct data sets to be produced
% k has to be smaller or equal 3


%contorl random number generation with seed
rng(RNG);
% generate data

A1=repmat([0.5 0.2; 0 2],N/2);
u1=repmat([4 0],1,(N/2));

A2=repmat([0.5 0.2; 0 0.3],N/2);
u2=repmat([5 7],1,(N/2));

A3=repmat([0.8 0; 0 0.8],N/2);
u3=repmat([7 4],1,(N/2));

bins=[round(M/k),2*round(M/k),3*round(M/k)];

data = randn(M,N) ;
for i=1:bins(1)
data(i,:)    = u1' + A1 * data(i,:)' ;
end
for i=(bins(1)+1):bins(2)
data(i,:)  = u2' + A2 * data(i,:)' ;
end
for i=(bins(2)+1):bins(3)
data(i,:) = u3' + A3 * data(i,:)' ;
end

end