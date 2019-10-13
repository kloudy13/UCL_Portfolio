%SL3 Q1 USED FOR WRITUP
clear all;
close all;

% Exercise 1 (Linear Example) Generate dataset X of 100 points xi of 10 dimensions,
% and whose components are unit variance Gaussian centered at the
% origin, and a 10-dimensional weight vector w from the same distribution. Generate
% the observations yi as hxi
% , wi + i, where the noise i is Gaussian with
% standard deviation 0.1.
% a) Split the data into 80 training and 20 testing points. Learn the weight
% vector using Ridge Regression in the primal form on the training data.
% Report the test error, and the difference between the learned and the true
% weight vectors.
% 1
% b) Apply Kernel Ridge Regression using the linear kernel (K(x, z) = hx, zi)
% and compare the results with a). Report the distance (norm) between
% the observations, the predictions in a) (primal) and the predictions in b)
% (dual).

%PART A
rng(55)
e_sd = 0.1;
x = mvnrnd(zeros(100,10), eye(10,10));

w = randn(10,1); %true weight

e = mvnrnd(zeros(100,1),e_sd^2); %error with sd 0.1
sd=std(e); %check

y =(x*w)+e;

%split the data
%here how you split doesnt matter but if had time dependent data then you
%need somehow shuffle data

x80 = x([1:80],:);
x20 = x([81:100],:);
y80 = y([1:80],:);
y20 = y([81:100],:);


%only additive model 
lambdafix = 0.01;
wcalc = RR(x80,y80,lambdafix); 
yhat_primal = (x20*wcalc); 

%Linear_Ridge_Regression_Model3

%test error

MSE_primal = mean(norm((y20-yhat_primal).^2));
 
[min_primal pos_min_primal] = min(MSE_primal);
wcalc_min = wcalc(:,pos_min_primal);

wdiff = abs(w-wcalc)./w; 

figure(1);
%plot distribution of x
%subplot(1,2,1);
plot(1:10,100*wdiff,'o')
title('Difference between true vs predicted weights (% of true)');
xlabel('Data dimension')
ylabel('Percent error, (%)')
axis([1 10 -25 25]);


grid on;
%grid major;



%PART B

%linear kernel RR:
%[alpha weight]=Linear_Kernel_Ridge_Regression_Model(x80',y80,kfix);
    
%[alphaKL weightKL]=KL_Linear_Kernel_Ridge_Regression_Model(x80,y80,lambdafix);

[alphaKRR weightKRR] = Kernel_RR('linear',x80,y80,lambdafix);
yhat_dual = zeros(size(x20,1),1);

for i = 1:size(x20,1)
    for j = 1:size(weightKRR,1)
        yhat_dual(i,1) = yhat_dual(i,1)+ (x20(i,:)*weightKRR(j,:)');
    end
end

MSE_dual = mean(norm((y20-yhat_dual).^2));
disp(MSE_primal);
disp(MSE_dual);

figure(2);
%subplot(1,2,2);
plot(81:100,y20,'.b')
hold on;
plot(81:100,yhat_primal,'xr');
plot(81:100,yhat_dual,'og');
title('True vs predicted labels (evaluated on the test set)');
xlabel('Data index');
ylabel('Y value (true, predicted primal, predictes dual');
legend('show');
legend('Ytrue','Yprimal','Ydual')
legend('Location','southwest')
%legend('boxoff')
axis([81 100 -5 5]);
grid on;
hold off;

