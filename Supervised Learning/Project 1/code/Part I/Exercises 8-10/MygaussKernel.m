% Exercse 10 part 3
function Kk = MygaussKernel(x1,x2,v)
% Gaussian Ridge Regression Kernel
% function calculates gaussian kernel using Eq.11 from assignmnet handout
% Input:
% x1 training data (x1 and x2 should have same no of columns)
% x2 validation data
% sigma is the variance parameter of gaussian kernel
% Output:
% Kernel matrix 

x = [x1; x2];

[a, ~] = size(x);

Kk = zeros(a,a);

for i=1:a
    for j=1:a
       Kk(i,j) = exp(-(norm(x(i,:)-x(j,:))^2)/(2*v^2));
    end
end
end