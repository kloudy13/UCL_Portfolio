function sol = winnowtrain(X,y)
% Function implements winnow algorithm to find weight vector
% Input:
% X is data matrix or vector and Y is vector (both should have same amount of rows)
% Output
% sol is a vector of the new weights 

% setup and initialise variables:
m = size(X,1);
n = size(X,2);
w = ones(n,1);

for t = 1:m
    currX = X(t,:)'; % invert working row of X matrix

    y_pred = (currX' * w >= n); % predicted y value 
    
    if y(t) ~= y_pred  % check if incorrect prediction  
        % if incorrect, calc new weight vector using winnow:
        w = w.* 2.^((y(t) - y_pred).*currX); 
    end
        
end

sol = w;

end





