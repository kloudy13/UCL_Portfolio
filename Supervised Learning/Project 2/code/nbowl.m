function z = nbowl(x,e)
m=size(x,1);
Ei=e*randn(m,1);
size(x);
z = zeros(m,1);
for i =1:m
    z(i,1)=x(i,1)^2+x(i,2)^2+Ei(i,1);
end
disp('bowl finished');