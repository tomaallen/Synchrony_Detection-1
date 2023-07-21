% Interface for training a GP using the SE Kernel
% 
% [mu, K] = gp_grad(X, Y, DY)
% 
% Input
% X: n by d matrix representing n training points in d dimensions
% Y: training values corresponding Y = f(X)
% Output
% mu: mean function handle such that calling mu(XX) for some predictive points XX calculates the mean of the GP at XX 
% K: dense kernel matrix

function [mu, K] = gp(X, Y, params)
[ntrain, ~] = size(X);

fprintf('SE with gradients: (ell, s, sigma1) = (%.3f, %.3f, %.3f)\n', ...
    exp(params.cov), params.sigma)

% Calculate interpolation coefficients
sigma2 = params.sigma^2*ones(1, ntrain);
K = se_kernel(X, params) + diag(sigma2);
lambda = K\Y;

% Function handle returning GP mean to be output
mu = @(XX) mean(XX, X, lambda, params);
end

function ypred = mean(XX, X, lambda, params)
KK = se_kernel(X, params, XX);
ypred = KK*lambda;

end