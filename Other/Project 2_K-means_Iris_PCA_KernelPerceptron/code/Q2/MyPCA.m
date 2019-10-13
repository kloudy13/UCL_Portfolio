%% PCA function

function [dataPC] = MyPCA(dataC,kp) 
% Purpose: perform Principle Component Analysis and return reduced data set
% Input: data, which needs to be centred and the number of new principal
% components (kp)
% Output: Reduced data set

    %%%% setup %%%%
    [M, ~] = size(dataC);
    %%%% test %%%% 
    if round(mean(dataC)) ~= 0
        error('Error: need to input centered data');
        return;
    end
    %%%% Calculate covariance matrix %%%%
    COV = (1/M)*(dataC'*dataC); %as per assignment hint
    %%%% Find eigensystem using Matlab command %%%%
    [vect, ~] = eig(COV); 
    % vector matrix in descending eigenvector order, such that need to extract 
    %the last kp columns of this matrix, which will form PC feature map. 

    %%%% Make feature map & reorder %%%%
    PCmap = vect(:, (end-kp+1):end);
    PCnew = zeros(size(PCmap));
        for i = 1:kp
            PCnew(:,i) = PCmap(:,kp+1-i);
        end
    PCmap = PCnew;
    %%%% Reduce data to PC form %%%%  
    dataPC = dataC*PCmap;
end
