%% Question 3
clear all;
close all;
rng(315)
%% Part A
% Generate data from a Gaussian Process with an exponential kernel
% Plot the result in a 3-D plot
m = 500; 
d = 2;
v = 0.5;
e_amp = 0;
miou = 0;
sigma = 1;

[Ya,xa,ka] = GP(m,d,v,e_amp,miou,sigma);
% standard 3-D graph
% figure(1);
% plot3(xa(:,1),xa(:,2),Ya,'.b');
% figure(1);
% imagesc(ka);
%Interpolated graph 3-D graph
% figure(2);
% xlin=linspace(min(xa(:,1)),max(xa(:,1)),100);
% ylin=linspace(min(xa(:,2)),max(xa(:,2)),100);
% [X,Y]=meshgrid(xlin,ylin);
% Z=griddata(xa(:,1),xa(:,2),Ya,X,Y,'cubic');
% 
% mesh(X,Y,Z); % interpolated
% axis tight; hold on
% plot3(xa(:,1),xa(:,2),Ya,'.','MarkerSize',15)
% hold off

%% Part b
m = 500; 
d = 10;
v0 = 5; %5
e_amp = 1;
miou = 0;
sigma_b = 1.3;%1.3
lambda = sigma_b^2;
av0 = (v0)*10.^(-2:0.5:2);
looplength = 100;
split1 = 0.126;
split2 = 0.252;
split3 = 1;
size_valid = round((split2-split1)*m);
n_bins = 50;

MSE_test = zeros(looplength,1);
best_v = zeros(looplength,1);
v_pos = zeros(looplength,1);
min_ecdf_dist = zeros(looplength,1);
MSE_valid = zeros(looplength,1);
normal_num = randn(size_valid,1);
yb_hat_valid = zeros(size_valid,1);

for loop = 1:looplength;
    
    % generate the data
    [Yb,xb,kb] = GP(m,d,v0,e_amp,miou,sigma_b);
       
    % split the data
    xbtrain = xb(1:split1*m,:);
    ybtrain = Yb(1:split1*m,:);
    xbvalid = xb(1+split1*m:split2*m,:);
    ybvalid = Yb(1+split1*m:split2*m,:); 
    xbtest = xb(1+split2*m:end,:);
    ybtest = Yb(1+split2*m:end,:);
    
    % construct loop for selection of vc
    ecdf_dist = inf*ones(length(av0),1);
    for vloop = 1:length(av0);
        v_current = av0(vloop);
        % perform ridge regression
        [~,yb_hat_valid] = km_krr(xbtrain,ybtrain,'gauss',v_current,lambda,xbvalid);
        Kb = km_kernel(xbtrain,xbtrain,'gauss',v_current);
        % make predictions for each value of av0(i)        
        sigma_i = ones(size(xbvalid,1),1);
        k_tmp = 0;
        k_i = zeros(size(xbtrain,1),1);
        for c = 1:size(xbvalid,1);
            for d = 1:size(xbtrain,1);
                k_i(d) = exp(-(norm(xbvalid(c,:)-xbtrain(d,:),2)^2)/(2*v_current^2)); 
            end
            % calculate the standard deviation of the prediction
            sigma_i(c,1) = 1 - k_i'*((Kb + lambda*eye(size(Kb)))\k_i);
        end        
        % calculate the prediction error
        err_i = ybvalid-yb_hat_valid;
        % calculate standardised error bars
        sd_err_i = (ybvalid-yb_hat_valid)./sqrt(sigma_i);
        % calcuate histogram
        hist_i = hist(sd_err_i,n_bins)/size_valid;
        % calculate eCDF
        eCDF_i = cumsum(hist_i);
        % compare to Gaussian CDF
        st_CDF = cumsum(hist(normal_num,n_bins)/size_valid);
        % add distance to counter
        ecdf_dist(vloop) = sum(norm(eCDF_i-st_CDF,2));
    end
    % select best v
    [min_ecdf_dist(loop), v_pos(loop)] = min(ecdf_dist);
    best_v(loop) = av0(v_pos(loop));

    % perform prediction on the test dataset with the best v found
    [alpha_i,yb_hat_test] = km_krr(xbtrain,ybtrain,'gauss',best_v(loop),lambda,xbtest);%best_v(loop),lambda
    
    % store the result 
     MSE_test(loop) = MSE(ybtest,yb_hat_test);
end

% display the results - MSE - v
disp('mean MSE: ');
mean(MSE_test)
disp('mean v: ');
mean(best_v)
% Plot MSE vs loop
figure(3);
plot(MSE_test);
title('MSE test error - based on 100points training set');
figure(4);
hist(MSE_test,20);
title('MSE test error - based on 20points training set');
% Plot hist of v
figure(5);
hist(best_v,10);
title('Selected v histogram');
% Plot predicted Y vs actual Y 
figure(6);
plot(ybtest,'b');
hold on; plot(yb_hat_test,'r');
hold off;
title('True (blue) vs estimated (red) Y values');
% 
figure(7);
hist(Yb);
title('Histogram of the true outputs (Yb)');
    
    

