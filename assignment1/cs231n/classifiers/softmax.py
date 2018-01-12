import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  print(W.shape)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = len(y)
  Y = np.zeros((num_train, num_classes))
  prediction = np.zeros_like(Y)
  Y[range(num_train), y] = 1.0
  for i in range(num_train):
    scores = np.exp(X[i, :].dot(W))
    prediction[i, :] = scores/np.sum(scores)
    #print(prediction[0:5])
    
    loss -= np.log(scores[y[i]]/ np.sum(scores))
    #loss += - np.sum(X[i, :].dot(W) * Y[i, :]) + np.log(np.sum(np.exp(X[i, :].dot(W))))
    dW += np.outer(X[i,:], prediction[i, :] - Y[i, :])
  loss = loss/num_train + reg * np.sum(W ** 2)
  dW = dW/num_train + 2 * reg * W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = len(y)
  Y = np.zeros((num_train, num_classes))
  Y[range(num_train), y] = 1.0
  prediction = np.zeros_like(Y)
  scores = np.exp(X.dot(W))
  scores = scores/ (np.sum(scores, axis = 1).reshape((num_train, 1)))
  correct_score = scores * Y
  S = np.max(correct_score, axis = 1).reshape(num_train, 1)

  #(scores[range(num_train), y].reshape((num_train, 1)))/ np.sum(scores, axis = 1)
  loss = np.sum(-np.log(S))
  #loss = np.sum(-np.log((S/np.sum(scores, axis = 1))))
  loss = loss / num_train + reg * np.sum(W ** 2)
  prediction = (np.exp(X.dot(W))) * ((1.0/np.sum(np.exp(X.dot(W)), axis = 1)).reshape((num_train,1)))
  dW = X.T.dot(prediction - Y)
  dW = dW / num_train + 2.0 * reg * W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

