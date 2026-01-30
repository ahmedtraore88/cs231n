from builtins import range
import numpy as np
from random import shuffle
#from past.builtins import xrange


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

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss
        p_copy = p.copy()
        p_copy[y[i]] -= 1
        product = np.outer(X[i], p_copy)
        dW += product 

    # normalize dw and add xorrectly L2 normalization
    dW = dW / num_train + 2 * reg * W

    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # DONE:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    N = X.shape[0]
    dW = np.zeros_like(W)


    #############################################################################
    # DONE:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    scores = X.dot(W)  # shape (N,C)
    scores -= np.max(scores, axis=1, keepdims=True)
    p = np.exp(scores)
    sum_values = np.sum(p, axis=1)
    p = p / sum_values[:, np.newaxis]  # to normalize
    logp = np.log(p)
    loss -= np.sum(logp[np.arange(N), y])  # negative log probability is the loss

    p_copy = p.copy()
    p_copy[np.arange(N), y] -= 1
    dW += X.T.dot(p_copy)

    # normalize dw and add xorrectly L2 normalization
    dW = dW / N + 2 * reg * W

    # normalized hinge loss plus regularization
    loss = loss / N + reg * np.sum(W * W)

    #############################################################################
    # DONE:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################


    return loss, dW
