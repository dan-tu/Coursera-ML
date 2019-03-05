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

%%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


% Feedforward each input to calculate hypothesis
a1 = [ones(m, 1), X];
a2 = [ones(m, 1), sigmoid(a1 * Theta1')];
hypothesis = sigmoid(a2 * Theta2');

% Vectorize y into an m x k matrix
vectorized_y = zeros(m, num_labels);
for i = 1:size(y)
    vectorized_y(i, y(i)) = 1;
end

% Calculate the cost
pos_case = -vectorized_y .* log(hypothesis);
neg_case = (1 - vectorized_y) .* log(1 - hypothesis);
cost_matrix = pos_case - neg_case;
J = sum(sum(cost_matrix)) / m;

% Add regularization to cost function

reg_theta1 = sum(sum(Theta1(:, 2:end).^2));
reg_theta2 = sum(sum(Theta2(:, 2:end).^2));
J = J + (reg_theta1 + reg_theta2) * lambda / (2 * m);
%%
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
BD_1 = zeros(num_labels, hidden_layer_size + 1); % First back prop
BD_2 = zeros(hidden_layer_size, input_layer_size + 1); % Second back prop
for i = 1:m
    % Feed forward
    input = [1, X(i, :)]; % 1 x 401
    a2 = [1, sigmoid(input * Theta1')]; % 1 x 26
    a3 = sigmoid(a2 * Theta2'); % 1 x k
    
    delta_3 = a3 - vectorized_y(i, :); % 1 x k. Error for each hypothesis
    delta_2 = delta_3 * Theta2 .* sigmoidGradient(a2); % 1x26. Error for each hidden node
    delta_2 = delta_2(2:end);

    BD_1 = BD_1 + (delta_3' * a2); % k x 26
    BD_2 = BD_2 + (delta_2' * input); % 25 x 401
end

Theta1_grad = BD_2 ./ m;
Theta2_grad = BD_1 ./ m;



%%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
