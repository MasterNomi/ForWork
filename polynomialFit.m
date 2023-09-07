% Generate synthetic data
rng(0); % Set random seed for reproducibility
u1 = rand(100, 1);
u2 = rand(100, 1);
u3 = rand(100, 1);
v1 = rand(100, 1);  % You can set these values as needed
v2 = rand(100, 1);  % You can set these values as needed
v = v1 .* exp(v2 .* u3);  % Compute v1 * e^(v2 * u3)
noise = 0.1 * randn(100, 1);
y = 2.0 * u1 .* u2 .* v + noise;  % True values of w1 and w2

% Fit polynomial regression model with u1, u2, and v1*v2*u3
X = [u1, u2, v1 .* v2 .* u3];  % Use u1, u2, and the new feature as input features
degree = 2; % Degree of polynomial
X_poly = [X, X.^degree]; % Add polynomial terms
mdl = fitlm(X_poly, y);

% Display model summary
disp(mdl);

% Plot predictions vs. actual
y_pred = predict(mdl, X_poly);
figure;
scatter(y, y_pred);
xlabel('Actual y');
ylabel('Predicted y');
title('Polynomial Regression with v1 * e^(v2 * u3)');
