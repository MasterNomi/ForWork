% Generate synthetic data based on your system's equation
rng(0); % Set random seed for reproducibility
u1 = rand(1000, 1);
u2 = rand(1000, 1);
u3 = rand(1000, 1);
noise = 0.1 * randn(1000, 1);
y = 2.0 * u1 .* u2 .* exp(0.5 * u3) + noise; % True values of w1 and w2

% Define the function that calculates the mean squared error
function mse = mean_squared_error(w)
    w1 = w(1);
    w2 = w(2);
    y_pred = w1 * u1 .* u2 .* exp(w2 * u3);
    mse = mean((y_pred - y).^2);
end

% Initial guess for w1 and w2
initial_guess = [1.0, 0.1];

% Perform optimization to find the optimal values of w1 and w2
options = optimoptions('fminunc', 'Display', 'iter', 'Algorithm', 'quasi-newton');
[optimal_w, mse_optimal] = fminunc(@mean_squared_error, initial_guess, options);

% Extract the optimal values of w1 and w2
optimal_w1 = optimal_w(1);
optimal_w2 = optimal_w(2);

fprintf('Optimal w1: %f\n', optimal_w1);
fprintf('Optimal w2: %f\n', optimal_w2);
fprintf('Optimal MSE: %f\n', mse_optimal);
