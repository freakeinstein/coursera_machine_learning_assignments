function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions = svmPredict(model, Xval);
error_old = mean(double(predictions ~= yval));

count = 0;
i=0.01;
do
  j=0.01;
  do
    count = count + 1;
    model = svmTrain(X, y, i, @(x1, x2) gaussianKernel(x1, x2, j));
    predictions = svmPredict(model, Xval);
    error_new = mean(double(predictions ~= yval));
    if (error_new < error_old) error_old = error_new; C = i; sigma = j; endif
    j = j * 3;
    disp(count)
  until (j>=30)
  i = i * 3;
  %disp(i)
until (i>=30)


% =========================================================================

end
