%% Question 1 Part 1.1 Implement K means Algorithm

function [y,centr] = MyKmeansAd2(X,k)
% Purpose: to classify vectors (data objects) into (k) classes by minimising 
% the Euclidean distance between centroids and object points
% Input: data matrix X and cluster size k
% Output: original matrix augmented by the group classification column (at the end)

[N M] = size(X);

c = zeros(k,M); % initialise matrix of centroids

% chose initial values for centroid:

p = randperm(N,k); 
% generate vector of random permutations with values from 1 to N, of length
% k. This will be use to index to data instances at random and initialise
% centroids.

for i = 1:k
    c(i,:) = X(p(i),:); %assign centroid values to random data points 
end

while (1)  %statement terminates when break statement below is  reached
          
  % Assignment step- assign each data point to closest cluster centre
  % done using MyDist2 function:
  [clust] = MyDist2(X,c);         % calculate object-centroid distances
                                % take minimum and corresponding cluster 
                                % thus make a new clustering.
                            
  % Update step- compute the new centroids
    NEWc = zeros(k,M);
    for ii = 1:k
        temp = X(clust == ii, :); %select data in category
        NEWc(ii, :) = mean(temp);
    end
    c=NEWc;
        
  % Recalculate new cluster using new centroid values:
  NEWclust = MyDist2(X,c);
        
  % while loop termination condition:
   if clust == NEWclust
            % solution converges: end while loop /stop iterating
            % this only occurs when all distances are minimised
      break;
   end
end

%output is original matrix augmented by the group classification column:
y = [X, clust];
centr=c;

end