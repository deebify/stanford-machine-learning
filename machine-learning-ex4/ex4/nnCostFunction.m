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

% -------------------------------------------------------------
    # The Cost Function "Without Regularization"
% Our Classes 
c1 = zeros(1,10);c1(1)=1;
c2 = zeros(1,10);c2(2)=1;
c3 = zeros(1,10);c3(3)=1;
c4 = zeros(1,10);c4(4)=1;
c5 = zeros(1,10);c5(5)=1;
c6 = zeros(1,10);c6(6)=1;
c7 = zeros(1,10);c7(7)=1;
c8 = zeros(1,10);c8(8)=1;
c9 = zeros(1,10);c9(9)=1;
c10 = zeros(1,10);c10(10)=1;

#implementing the FeedForward 

% Add bias to Layer 1
X = [ones(size(X)(1),1) X];
%  Layer 1 
Z1 = X*Theta1'; 
% Layer 1 After activaion
A1 = sigmoid(Z1);

%-----------------

% Add bias to Layer 2 
A1 = [ones(size(A1)(1),1) A1];
% Layer 2
Z2 = A1*Theta2';
% Layer 2 After activation 
A2 = sigmoid(Z2);


hypoth = A2; 
C = zeros(m,10);

% Calculate the Cost Function 
for i=1:m

  if(y(i) == 1)
    C(i,:) = -c1 .* log(hypoth(i,:)) - (1-c1) .* log(1-hypoth(i,:));
  end
  if(y(i) == 2)
    C(i,:) = -c2 .* log(hypoth(i,:)) - (1-c2) .* log(1-hypoth(i,:));
  end
  if(y(i) == 3)
    C(i,:) = -c3 .* log(hypoth(i,:)) - (1-c3) .* log(1-hypoth(i,:));
  end
  if(y(i) == 4)
    C(i,:) = -c4 .* log(hypoth(i,:)) - (1-c4) .* log(1-hypoth(i,:));
  end
  if(y(i) == 5)
    C(i,:) = -c5 .* log(hypoth(i,:)) - (1-c5) .* log(1-hypoth(i,:));
  end
  if(y(i) == 6)
    C(i,:) = -c6 .* log(hypoth(i,:)) - (1-c6) .* log(1-hypoth(i,:));
  end
  if(y(i) == 7)
    C(i,:) = -c7 .* log(hypoth(i,:)) - (1-c7) .* log(1-hypoth(i,:));
  end
  if(y(i) == 8)
    C(i,:) = -c8 .* log(hypoth(i,:)) - (1-c8) .* log(1-hypoth(i,:));
  end
  if(y(i) == 9)
    C(i,:) = -c9 .* log(hypoth(i,:)) - (1-c9) .* log(1-hypoth(i,:));
  end
  if(y(i) == 10)
    C(i,:) = -c10 .* log(hypoth(i,:)) - (1-c10) .* log(1-hypoth(i,:));
  end

end

J = (1/m)*sum(sum(C,2));


% -------------------------------------------------------------
 # Add the Regularization Part to Cost Function 
 
 
 
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
