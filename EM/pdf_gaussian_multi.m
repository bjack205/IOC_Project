function [ p ] = pdf_gaussian_multi(x,mu,sigma)
% PDF for multivariate Gaussian
n = length(x);
p = 1/(2*pi*det(sigma)^(1/2))*exp(-0.5*(x-mu)'*inv(sigma)*(x-mu));
end

