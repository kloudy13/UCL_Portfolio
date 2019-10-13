function [alpha weights] = Kernel_RR(kernel,X,y,lambda,v)
% Estimate Kernel Ridge Regression Parameters
% Inputs:
%           Kernel type
%           Training data matrix, X
%           Training labels vector, y
%           kernel width, v (for Gaussian kernel)

[m,d] = size(X);

if nargin < 5;
    v = 1;
end

switch kernel
    
    case 'linear';     
       alpha = (X*X' + lambda * eye(m,m))\y;
       weights = diag(alpha)*X ;   
    
    case 'quadratic';  
        C = zeros(m,m);
        for i = 1:m
            for j=1:m
                C(i,j) = quadratic_kernel(X(i,:),X(j,:),v);
            end
        end
%         disp('X:')
%         size(X)
        alpha = ( C + lambda * eye(m,m))\y;
%         disp('alpha:')
%         size(alpha)
        weights = diag(alpha)*X ;   
%         size(alpha);
%         size(X);    
    
    case 'gauss';      
        for i = 1:m
            for j=1:m
                C(i,j) = gauss_kernel(X(i,:),X(j,:),v);
            end
        end
        alpha = ( C+  lambda *eye(m,m))\y;
        size(alpha);
        size(X);
        weights = diag(alpha)*X ;
        
end
    



