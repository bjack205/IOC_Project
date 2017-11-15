% Taylor Howell
% EM algorithm to fit mixture of Gaussians
% 11-14-2017

clear;clc;close all;
n = 2; % dimension of each training example
k = 4; % number of clusters

%% Generate k 2-dimensional Gaussians and sample from them to create training examples
examples = 100;
mu_initial = [];
sigma_initial = [];
Xi = [];
X = [];
for i = 1:k
    mu_initial{i} = (i^2)*ones(1,n)
    sigma_initial{i} = 2*rand(1)*eye(n);
    Xi{i} = randn(examples, n) * chol(sigma_initial{i}) + repmat(mu_initial{i}, examples, 1);
    X = cat(1,X,Xi{i});
end

%% Run EM
[mu, sigma] = EM_mix_gauss(X,k);

%% Visualize Mixture of Gaussians
gridSize = 100;
u = linspace(-5, 20, gridSize);
[A B] = meshgrid(u, u);
gridX = [A(:), B(:)];

figure
subplot(1,2,1)
hold on;
for j = 1:k
    % plot raw data
    plot(Xi{j}(:, 1), Xi{j}(:, 2), 'o');
    for i = 1:size(gridX,1)
        z(i) = pdf_gaussian_multi(gridX(i,:)', mu_initial{j}', sigma_initial{j});
    end
    % plot contour from know distributions
    contour(u, u, reshape(z, gridSize, gridSize),'k');
end
title('Fit from know distributions')
axis square

subplot(1,2,2)
hold on;
for j = 1:k
    % plot raw data
    plot(Xi{j}(:, 1), Xi{j}(:, 2), 'o');
    for i = 1:size(gridX,1)
        z(i) = pdf_gaussian_multi(gridX(i,:)', mu(j,:)', sigma{j});
    end
    
    % plot contour from know distributions
    contour(u, u, reshape(z, gridSize, gridSize),'k');
end
title('Mixture of Gaussians')
axis square