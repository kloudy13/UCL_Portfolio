%% Question 2
clear all;
close all;
%% Part A
% Generate 1000 points from a 2-D quadratic gaussian process with quadratic kernel
rng(119)
d = 2;
m = 5000;
x = mvnrnd(zeros(m,d), eye(d,d)); 
r = zeros(m,d,d);
z = zeros(m,d^2);
sigma_a = 1;
for i = 1:m
    r(i,:,:) = x(i,:)'*x(i,:);
    z(i,:) = reshape(r(i,:,:),[d^2,1]);
end

w = randn(d^2,1);
e = mvnrnd(zeros(m,1), sigma_a);

ya = (z*w);

figure(1);
plot3(x(:,1),x(:,2),ya,'o')
title('3-D plot fo data from a quadratic process, n=5000, d=2, sigma=1');
xlabel('Dimension 1')
ylabel('Dimension 2')
zlabel('Y values')
grid on;


%% PART B
d = 20;
m = 500;
xb = mvnrnd(zeros(m,d), eye(d,d)); 
rb = zeros(m,d,d);
zb = zeros(m,d^2);
sigma_b = 5;
lambda_b = sigma_b^2;
split_b = 0.25;

for i = 1:m
    rb(i,:,:) = xb(i,:)'*xb(i,:);
    zb(i,:) = reshape(rb(i,:,:),[d^2,1]);
end
wb = randn(d^2,1);
eb = mvnrnd(zeros(m,1), sigma_b^2);

yb = (zb*wb)+eb;

xbtrain = xb(1:split_b*m,:);
ybtrain = yb([1:split_b*m],:);
xbtest = xb(1+split_b*m:end,:);
ybtest = yb(1+split_b*m:end,:);


% Run Ridge Regression with correct lambda and report test error

[alpha_b weights_b] = Kernel_RR('quadratic',xbtrain,ybtrain,lambda_b,1); 

for i = 1:size(xbtest,1);
    yhat_b(i,1) = 0;
    for j = 1:size(xbtrain,1);
        yhat_b(i,1) = yhat_b(i,1) + alpha_b(j)*quadratic_kernel(xbtest(i,:),xbtrain(j,:));
    end
end
mse_b_test = MSE(ybtest,yhat_b)

%% PART C

loop=100;
MSE100=zeros(loop,1);
selectAlphaC=zeros(loop,1);
d = 20;
m = 500;
sigma_c = 5;
splitc1 = 0.25;
splitc2 = 0.4;
splitc3 = 1;
lambda_c = (sigma_c^2)*logspace(-4,4,33)';
minMSEvalid = zeros(loop,1);
pos = zeros(loop,1);
wc = randn(d^2,1);

for j = 1:loop
    % produce the data   
    xc = mvnrnd(zeros(m,d), eye(d,d)); 
    rc = zeros(m,d,d);
    zc = zeros(m,d^2);
    for i = 1:m
        rc(i,:,:) = xc(i,:)'*xc(i,:);
        zc(i,:) = reshape(rc(i,:,:),[d^2,1]);
    end
    
    ec = mvnrnd(zeros(m,1), sigma_c^2);
    yc = (zc*wc)+ec;
    % split to train, validation and test set
    xctrain = xc(1:splitc1*m,:);
    yctrain = yc([1:splitc1*m],:);
    xcvalid = xc(1+splitc1*m:splitc2*m,:);
    ycvalid = yc([1+splitc1*m:splitc2*m],:); 
    xctest = xc(1+splitc2*m:end,:);
    yctest = yc(1+splitc2*m:end,:);
    
    % run KRR for the various lambda values
    yc_hat_valid = zeros(size(ycvalid));
    mse_c_valid = size(lambda_c,1);
    for l = 1:size(lambda_c,1)
        
        [alpha_c weights_c] = Kernel_RR('quadratic',xctrain,yctrain,lambda_c(l),1); 
        
        for ll = 1:size(xcvalid,1);
            yc_hat_valid(ll,1) = 0;
            for lll = 1:size(xctrain,1);
                yc_hat_valid(ll,1) = yc_hat_valid(ll,1) + alpha_c(lll)*quadratic_kernel(xcvalid(ll,:),xctrain(lll,:));
            end
        end

        mse_c_valid(l) = MSE(ycvalid,yc_hat_valid);
    end
    
    [minMSEvalid(j) pos(j)]=min(mse_c_valid);
    
    % re-run KRR with the selected lambda value and predict on the test set
    
    [alpha_c weights_c] = Kernel_RR('quadratic',xctrain,yctrain,lambda_c(pos(j)),1);
    
    for ll = 1:size(xctest,1);
        yc_hat_test(ll,1) = 0;
        for lll = 1:size(xctrain,1);
            yc_hat_test(ll,1) = yc_hat_test(ll,1) + alpha_c(lll)*quadratic_kernel(xctest(ll,:),xctrain(lll,:));
        end
    end
       
    MSE100(j) = MSE(yctest,yc_hat_test);
    
    selectAlphaC(j,1) = lambda_c(pos(j),1);
    disp(j)
end

meanMSE100 = mean(MSE100);
meanselectAlphaC100 = mean(selectAlphaC);

figure(2);
plot(MSE100);
title('Least Mean Square Test Error for the 100 trials');
xlabel('Trial number');
ylabel('Calculated MSE with (best) selected lambda');

figure(3);
hist(pos,20);
title('Frequency histogram of the index of the selected lambda value during the 100-cycle validation');
xlabel('Index value (bin 17 corresponds to the theoretical values)');
ylabel('Number of occurances');

figure(4);
plot(yctest,'g');
hold on;
plot(yc_hat_test,'r');
hold off;
title('True v.s. predicted y values for a typical data-set');
xlabel('Index of data point');
ylabel('Y value');
legend('show');
legend('Ytrue','Ypredicted')
legend('Location','southwest')
