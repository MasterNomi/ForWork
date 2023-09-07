% Generate synthetic data
rng(0); % Set random seed for reproducibility
u1 = rand(100, 1);
u2 = rand(100, 1);
u3 = rand(100, 1);
noise = 0.1 * randn(100, 1);
y_true = 2.0 * u1 .* u2 .* (0.5 * exp(-0.2 * u3)) + noise; % True values of coefficients

% Create a neural network model
net = feedforwardnet([10, 10]); % Define the neural network architecture
X = [u1, u2, u3]'; % Input features
y_true = y_true'; % Output
net = train(net, X, y_true);

% Test the model on the training data
y_pred = net(X);

% Display the model architecture and performance
view(net);
figure;
scatter(y_true, y_pred);
xlabel('Actual y');
ylabel('Predicted y');
title('Neural Network Model');
