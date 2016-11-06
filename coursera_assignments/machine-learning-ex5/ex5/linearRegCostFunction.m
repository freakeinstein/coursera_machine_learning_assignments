function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Cost calculation
  % without regularization
  htx = X * theta;
  dist = htx - y;
  sqerror = dist .^ 2;
  error_tot = sum(sqerror);
  J = error_tot/(2*m);
  %with regularization
  reg = sum(theta(2:end,:) .^ 2);
  reg = reg * lambda / (2*m);

  J = J + reg;
  
 % Gradient calculation
  pre = dist .* X;
  unreg = sum(pre)/m;
  
  % before regularization, we need modified theta by eliminating first one
  mod_theta = [0;theta(2:end,:)];
  mod_theta = mod_theta .* (lambda/m);
  
  grad = unreg' .+ mod_theta;




% =========================================================================

grad = grad(:);

end
