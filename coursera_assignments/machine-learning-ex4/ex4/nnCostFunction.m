function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% PART 1:
X = [ones(m, 1) X];
z1 = X * Theta1';
h = sigmoid(z1);
H = [ones(m, 1) h];
z2 =  H* Theta2';
P = sigmoid(z2);


% lets modify y as R^(m x k)
my_y = -1 .* bsxfun(@eq,y,1:num_labels);

% lets compute cost
left = my_y .* log(P);
right = ((1 .+ my_y) .* log(1 .- P));
J = sum(sum((left - right),2)) / m;

%lets add regularization

J = J + ( sum(sum( (Theta1(:,2:size(Theta1,2)) .^ 2) ,2)) + sum(sum( (Theta2(:,2:size(Theta2,2)) .^ 2) ,2)) ) * ( lambda / (2*m) );



% PART 2: [ NOT OK ! ]

% finding errors on each layer
d_O = P + my_y;
%dsig_O = sigmoidGradient(z2);
d_EO = d_O;% .* dsig_O;

d_H = d_EO * Theta2;
d_h = d_H(:,2:size(d_H,2));
dsig_h = sigmoidGradient(z1);
d_Eh = d_h .* dsig_h;

% finding gradients of each parameters 
Theta2_grad = ( d_EO' * H ) ./ m;
Theta1_grad = ( d_Eh' * X ) ./ m;

Theta1_grad = Theta1_grad + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
Theta2_grad = Theta2_grad + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
