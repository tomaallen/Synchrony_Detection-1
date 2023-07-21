clear all

load('data\CF 2022\training_data.mat')
load('data\CF 2022\val_data.mat')
load('data\CF 2022\test_data.mat')

X=[training_input_num];
Y=[training_label_num];
X_val=[val_input_num];
Y_val_truth=[val_label_num];

% Initial hyperparameters
ell0 = 0.5*sqrt(size(X,2));
s0 = std(Y);
sig0 = 5e-2*s0; 
beta = 1e-6;
sigma = sqrt(exp(2*log([sig0])) + beta);
params = struct('cov', log([ell0, s0]), 'lik', log([sig0]), 'sigma', sigma);

[mu,K]=gp(X,Y,params);
Y_val=mu(X_val);



figure
plot(Y_val,'o')
hold on
plot(Y_val_truth, '<')
hold off


X_test = test_input_num;
Y_test=mu(X_test);
Y_test_truth=test_label_num;
figure
plot(Y_test,'o')
hold on
plot(Y_test_truth, '<')
hold off

if 0
figure
plot(Y,'o')
hold on
Y_gp=mu(X);
plot(Y_gp, '<')
hold off
end


