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

%%
% Generate synthetic data
rng(0); % Set random seed for reproducibility
u1 = rand(100, 1);
u2 = rand(100, 1);
u3 = rand(100, 1);
noise = 0.1 * randn(100, 1);
y_true = 2.0 * u1 .* u2 .* (0.5 * exp(-0.2 * u3)); % True values of coefficients

% Create a function that represents the model equation
model = @(coefficients, X) coefficients(1) * X(:, 1) .* X(:, 2) .* (coefficients(2) * exp(coefficients(3) * X(:, 3)));

% Define an initial guess for the coefficients
initial_guess = [1, 1, 1];

% Define a loss function (e.g., mean squared error)
loss_function = @(coefficients) mean((y_true - model(coefficients, [u1, u2, u3])).^2);

% Optimize the coefficients using a numerical optimization method
options = optimset('Display', 'iter'); % Adjust optimization options as needed
estimated_coefficients = fminsearch(loss_function, initial_guess, options);

% Extract the estimated coefficients
w1 = estimated_coefficients(1);
v1 = estimated_coefficients(2);
v2 = estimated_coefficients(3);

% Display the estimated coefficients
disp(['Estimated w1: ', num2str(w1)]);
disp(['Estimated v1: ', num2str(v1)]);
disp(['Estimated v2: ', num2str(v2)]);

