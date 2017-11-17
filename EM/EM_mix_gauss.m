function [ mu, sigma ] = EM_mix_gauss( X,k,eps )
% Implementation of the EM algorithm to fit a mixture of Gaussians.

% X is all training examples (one training example per row). k is the
% number of Gaussians x is theorized to be drawn from. eps is the
% convergence criteria.
[m,n] = size(X);

% initialize k cluster centroids randomly from the data
mu = datasample(X,k);

% initialize covariance matrix for each cluster as covariance of all
% training examples
sigma = [];
for j = 1:k
    sigma{j} = cov(X);
end

% initialize equal priors
phi = ones(1,k)./k;

% initialized weights: w_j = p(z = j| x)
W = zeros(m,k);

iteration = 0;
error = 1;

while error > eps
    fprintf('Iteration: %d\n',iteration);
    iteration = iteration + 1;

% E-step
    p = zeros(m,k);
    pw = zeros(m,k);

    for j = 1:k
        p(:,j) = pdf_gaussian_multi(X,mu(j,:),sigma{j});
        pw(:,j) = p(:,j)*phi(j);
    end

    W = pw./repmat(sum(pw,2),1,k);

% M-step
    mu_old = mu;

    for j = 1:k
        % recalculate priors
        phi(j) = mean(W(:,j),1);

        % recalculate means
        mu(j,:) = (W(:,j)'*X)./sum(W(:,j),1);

        sigma_k = zeros(n,n);

        for i = 1:m
            sigma_k = sigma_k + (W(i,j).*(X(i,:)-mu(j,:))'*(X(i,:)-mu(j,:)));
        end
        sigma{j} = sigma_k./sum(W(:,j));
    end
    error = sum((mu-mu_old).^2);
end
end

