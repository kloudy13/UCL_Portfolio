function [Y,x,k] = GP(m,d,v,e_amp,miou,sigma);
% Generate m (d-dimesnional) data points from a Gaussian Process 
% with Radial Function of width v
    x = mvnrnd(zeros(m,d), eye(d,d));
    u = mvnrnd(zeros(m,1),1);
    
    k = zeros(m,m);
    for i = 1:m;
        for j = 1:m;
            k(i,j) = exp(-(norm(x(i,:)-x(j,:),2)^2)/(2*v^2));
        end
    end
    
    L = chol(k+0.0000000001*eye(m,m),'lower');
    
    if e_amp>0;
        e = mvnrnd(miou*ones(m,1),sigma^2);
        Y = L*u + e;
    else
        Y = L*u; 
    end
    
    