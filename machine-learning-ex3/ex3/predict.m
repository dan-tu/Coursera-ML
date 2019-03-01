function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add column of 1's to input X
X = [ones(m, 1), X]; % m x 401

% Calculate hidden layer node values
a2 = sigmoid(Theta1 * X'); % Gives us a 25 x m matrix

% Calculate the output layer
a2 = [ones(1, m); a2]; % Add bias input to hidden layer. 26 x m
a3 = sigmoid(Theta2 * a2); % Gives us a 10 x m matrix of prob for each class

% Extract the highest probability class for each set
[~, p] = max(a3);
p = p';

% =========================================================================


end
