# -*- coding: utf-8 -*-
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # loss function을 W에 대해서 미분한 것. 
  # 즉, max(sj - syi + 1) 을 미분한 것과 동일하다고 볼 수 있음.

  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  for i in xrange(num_train):
    scores = X[i].dot(W) #scores의 shape는 (C,)
    correct_class_score = scores[y[i]] #정답에 해당하는 correct score. shape는 (1, )
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i] #j번째 클래스 스코어의 미분값 더하기
        dW[:, y[i]] -=  X[i] #i번째 정답 클래스 yi의 미분값 빼기
        #어차피 1은 상수값이기 때문에 사라짐.

  loss /= num_train
  dW /= num_train

  # Add L2 regularization to the loss. (오버피팅 방지)
  loss += reg * np.sum(W * W)
  dW += 2*reg*W 
  #L2 regularization을 W에 대해서 미분한 값을 추가하기

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = np.dot(X, W)
  correct_class_score = scores[range(num_train), y]
  margins = np.maximum(0, scores - np.reshape(correct_class_score, [num_train, 1]) + 1)
  margins[range(num_train), y] = 0

  loss = np.sum(margins)
  loss /= num_train
  loss += reg*np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  margins[margins > 0] = 1
  valid_margin_count = margins.sum(axis=1)
  # Subtract in correct class (-s_y)
  margins[range(num_train),y] -= valid_margin_count
  dW = np.dot(X.transpose(), margins)
  dW /= num_train
  dW += 2*reg*W 

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
