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
  scores = np.zeros((X.shape[0],W.shape[1]))
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i  in xrange(0,X.shape[0]):
    scores[i,:] = X[i].dot(W)
    scores[i,:] -= np.max(scores)
    
    scores[i,:] = np.exp(scores[i,:])
    scores[i,:] /= np.sum(scores[i,:])
  
    #loss -= np.sum(scores[i,:])
    #loss -= np.sum(np.log(scores[i,:]))
  #print(scores[0,:])
  loss -= np.sum(np.log(scores[np.arange(X.shape[0]), y]))
  loss /= X.shape[0]
  loss += reg * np.sum(W**2)
  #print(loss)
  scores[np.arange(X.shape[0]), y] -= 1
  dW = np.dot(X.T, scores)
  dW /= X.shape[0]
  dW += reg*W 
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
  scores = np.dot(X,W)
  #scores -= np.max(scores)
  scores = np.exp(scores)
  scores /= np.sum(scores, axis=1, keepdims=True)
  
  loss -= np.sum(np.log(scores[np.arange(X.shape[0]), y]))
  loss /= X.shape[0]
  loss += reg * np.sum(W**2)
  #print([np.arange(X.shape[0]), y])
  scores[np.arange(X.shape[0]), y] -= 1
  #print(scores[0,:])
  dW = np.dot(X.T, scores)
  dW /= X.shape[0]
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

