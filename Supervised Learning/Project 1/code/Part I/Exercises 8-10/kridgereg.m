function out = kridgereg(K,y,gamma)
% preform kernel ridge regression as per Eq.12 (assignment handout)
% Input:
% K is the kernel matrix - should be square!
% y is a vector of the dependant variable 
% gamma is the scalar ridge parameter
% Output:
% Out == alpha vector of dual weights

len = size(K,1);
    if len ~= size(K,2) % check if K is square 
        disp('Kernel matrix shoudl be square')
    end    
    
out = (K + gamma * len * eye(len)) \ y; 

end