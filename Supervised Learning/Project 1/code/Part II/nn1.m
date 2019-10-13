function loss = nn1(trainX, trainY, testX, testY)
% Function implemnets k nearest neighbour algorithm for k=1
% inputs:
% X matrix of data test/ train respectively 
% Y vector of dependent variable test/ train respectively 

% initialise variables 
loss = 0; % count for the generalisation error
startSize = size(trainX,1); 
m = size(testX,1);
% create matrix to store nearest neighbour distances: 
distance=zeros(startSize,m); 

for t = 1:m
    for j=1:startSize
    % find distances between train and test points
    distance(j,t) = sum((trainX(j,:) - testX(t,:)).^2, 2);
    end
    % find the index of minimum distance  
    % (corresponidng to the nearest training/test point pair) 
    [~, index] = min(distance(:,t));
    
    % set predicted Y to the distance minimising Y:
    yP = trainY(index);
    
    % find loss by counting miss-match between predicted Y and test Y:
    if yP ~= testY(t)
        loss = loss + 1;
    end
end

end