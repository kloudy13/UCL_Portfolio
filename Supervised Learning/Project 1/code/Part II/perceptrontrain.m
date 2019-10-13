function alpha = perceptrontrain(X,y,d)
% Function implements perception algorithm
% Input:
% X is data matrix and Y is vector (both should have same amount of rows).
% Also, y must be between -1 and 1.
% d is the kernel parameter
% Output
% vector of weights parameters (alpha) 

% initialisation
alpha= zeros(size(X,1),1);
mistake = true;
kernel = (X*X').^d;

if mistake == true
    mistake = false;
    for i = 1: size(X,1) 
        currentsum = alpha'* kernel(i,:)';
        y_pred = sign(currentsum); % find sign for prediction

        % update: 
        if y_pred ~= y(i)  % check if incorrect prediction  
        % if incorrect, calc new alpha:
            alpha(i,1) = alpha(i,1) + y(i);
            mistake = true;
        end 
    end
end
    
        

        
        
    
    