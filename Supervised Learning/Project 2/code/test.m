rng(15);
% Generate test data
[X1 X2]=meshgrid(-1.9:.1:2,-1.9:.1:2);
x1 = reshape(X1,[size(X1,2)^2,1]);
x2 = reshape(X2,[size(X2,2)^2,1]);
X=[x1,x2];
z = nbowl(X,0.1);

% get the randomly-selected indices
lx = size(X,1);
indices = randperm(lx);
% choose the subset of a you want
split = 0.75;
numelements = round(split*lx);
train_idx = indices(1:numelements);
test_idx = indices(1+numelements:end);
x_train = X(train_idx,:);
x_test = X(test_idx,:);
z_train = z(train_idx,:);
z_test = z(test_idx,:);

% Get regression coefficients
[a_RR w_RR] = Kernel_RR('quadratic',x_train,z_train,0.01,1);
disp('size a_i')    
size(a_RR)
disp('size x_test')    
size(x_test)
for i = 1:size(x_test,1);
    y_hat(i,1) = 0;
    for j = 1:size(x_train,1);
        y_hat(i,1) = y_hat(i) + a_RR(j)*quadratic_kernel(x_test(i,:),x_train(j,:));
    end
end
disp('size y_hat')    
size(y_hat)

figure(1)
plot3(x_train(:,1),x_train(:,2),z_train,'g.')
hold on;
plot3(x_test(:,1),x_test(:,2),y_hat,'r.')
hold off;


figure(1);
surf(X1,X2,reshape(z,40,40));
hold on;
plot3(x_test(:,1),x_test(:,2),y_hat,'r*');
hold off;
