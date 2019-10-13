%rng(315);
m=2000;
d=1;
v0=0.05;
e_amp = 1;
miou = 0;
sigma_b = 1.3*10^-5;
split1 = 0.126;
lambda = sigma_b^2;

% generate the data
[Yb,xb,kb] = GP(m,d,v0,e_amp,miou,sigma_b);
% split the data
xbtrain = xb(1:split1*m,:);
ybtrain = Yb([1:split1*m],:);

xbtest = xb(1+split1*m:end,:);
ybtest = Yb(1+split1*m:end,:);
v=5;
% perform ridge regression
[alpha_i,yb_hat_test] = km_krr(xbtrain,ybtrain,'gauss',v0,lambda,xbtest);

figure(1)
plot(xbtest,ybtest,'ob')
hold on;
plot(xbtest,yb_hat_test,'xr')
hold off
figure(2)
plot(ybtest,'-xb')
hold on;
plot(yb_hat_test,'-r')
hold off
axis([0 100 -3 3]);