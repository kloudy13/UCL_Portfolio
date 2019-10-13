function K = kernel(X, Xt, d,type)
% KERNEL the kernel function
% usage: [K] = kernel(X, Xt, d, type)
% KERNEL type
switch type
 case 'polynomial'
% polynomilal
K = (X*Xt').^d;
  case 'gauss'
% gauss
  n= size(X,1);
  m = size(Xt,1);
  s = sum(X.^2,2);
  st = sum(Xt.^2,2);
  mtt = repmat(s,1,m);
  mttt = repmat(st',n,1);
  sq = mtt + mttt - 2*X*Xt'; 
  K = exp(-d*(sq));
end
