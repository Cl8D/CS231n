# -*- coding: utf-8 -*-
from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.
    (즉, FC- relu- FC - softmax 이런 순서!)

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        #weight, bias 초기화 
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)

        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        N = X.shape[0]
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        #layer_utils.py에 정의되어 있는 affine_relu_forward 함수와
        #layer.py에 정의되어 있는 affine_forward 함수를 사용한다.
        fc1, cache1 = affine_relu_forward(X, W1, b1)
        scores, cache2 = affine_forward(fc1, W2, b2)                

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        #layers.py에 정의되어 있는 softmax_loss 함수
        data_loss, dscores = softmax_loss(scores, y) #loss, d_scores의 값

        reg_loss = 0.5 * self.reg * np.sum(W1 * W1) + 0.5 * self.reg * np.sum(W2 * W2)
        loss = data_loss + reg_loss

        #compute the gradients
        dh1, grads['W2'], grads['b2'] = affine_backward(dscores, cache2)
        _, grads['W1'], grads['b1'] = affine_relu_backward(dh1, cache1)

        #Add the regularization gradient
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    #model 순서
    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        #일종의 초기화 과정.
        layers_dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            wi = 'W' + str(i+1)
            bi = 'b' + str(i+1)

            self.params[wi] = weight_scale * np.random.randn(layers_dims[i], layers_dims[i + 1])
            self.params[bi] = np.zeros(layers_dims[i + 1])
            if self.use_batchnorm and i < len(hidden_dims):
              gai = 'gamma' + str(i+1)
              bei = 'beta' + str(i+1)
              self.params[gai] = np.ones(layers_dims[i + 1])
              self.params[bei] = np.zeros(layers_dims[i + 1])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        h, cache1, cache2, cache3, cache4, bn, out = {}, {}, {}, {}, {}, {}, {}
        out[0] = X

        # Forward pass: compute loss
        for i in range(self.num_layers - 1):
            W = self.params['W' + str(i + 1)]
            b = self.params['b' + str(i + 1)]
            
            if self.use_batchnorm:
                gamma = self.params['gamma' + str(i + 1)]
                beta = self.params['beta' + str(i + 1)]

                h[i], cache1[i] = affine_forward(out[i], W, b)
                bn[i], cache2[i] = batchnorm_forward(h[i], gamma, beta, self.bn_params[i])
                out[i + 1], cache3[i] = relu_forward(bn[i])
                
                if self.use_dropout:
                    out[i + 1], cache4[i] = dropout_forward(out[i + 1], self.dropout_param)
            
            else:
                out[i + 1], cache3[i] = affine_relu_forward(out[i], W, b)
                if self.use_dropout:
                    out[i + 1], cache4[i] = dropout_forward(out[i + 1], self.dropout_param)
        
        W = self.params['W' + str(self.num_layers)]
        b = self.params['b' + str(self.num_layers)]
        scores, cache = affine_forward(out[self.num_layers - 1], W, b)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        loss, reg_loss, grads = 0.0, 0.0, {}
        data_loss, dscores = softmax_loss(scores, y)

        for i in range(self.num_layers):
          W = self.params['W' + str(i + 1)]
          reg_loss += 0.5 * self.reg * np.sum(W * W)

        loss = data_loss + reg_loss

        # Backward pass: compute gradients
        dout, dbn, dh, ddrop = {}, {}, {}, {}
        t = self.num_layers - 1
        dout[t], grads['W' + str(t + 1)], grads['b' + str(t + 1)] = affine_backward(dscores, cache)
        
        for i in range(t):
          if self.use_batchnorm:
              if self.use_dropout:
                ddrop[t - 1 - i] = dropout_backward(dout[t - i], cache4[t - 1 - i])
                dout[t - i] = ddrop[t - 1 - i]
              
              dbn[t - 1 - i] = relu_backward(dout[t - i], cache3[t - 1 - i])
              dh[t - 1 - i], grads['gamma' + str(t - i)], grads['beta' + str(t - i)] = batchnorm_backward(dbn[t - 1 - i], cache2[t - 1 - i])
              dout[t - 1 - i], grads['W' + str(t - i)], grads['b' + str(t - i)] = affine_backward(dh[t - 1 - i],cache1[t - 1 - i])
            
          else:
            if self.use_dropout:
              ddrop[t - 1 - i] = dropout_backward(dout[t - i], cache4[t - 1 - i])
              dout[t - i] = ddrop[t - 1 - i]
              
            dout[t - 1 - i], grads['W' + str(t - i)], grads['b' + str(t - i)] = affine_relu_backward(dout[t - i], cache3[t - 1 - i])

        # Add the regularization gradient contribution
        for i in range(self.num_layers):
            grads['W' + str(i + 1)] += self.reg * self.params['W' + str(i + 1)]
        
    
        '''
        loss, dscores = softmax_loss(scores, y)
        
        for i in range(self.num_layers, 0, -1) :
          wi = 'W' + str(i)
          bi = 'b' + str(i)
          loss += 0.5 * self.reg * np.sum(np.square(self.params[wi]))

          if i == self.num_layers: #마지막이라면 loss를 바라본다...?
            dout, grads[wi], grads[bi] = affine_backward(dscores, caches[i])

          else:
            if self.use_dropout:
              dout = dropout_backward(dout, caches['dropout' + str(i)])

              #batch norm을 사용하거나, 아니거나.      
              if self.normalization == 'batchnorm' or self.normalization == 'layernorm':     
                  fc_cache, norm_cache, relu_cache = caches[i]
                  dout = relu_backward(dout, relu_cache)
                    
                  if self.normalization == 'batchnorm':
                    dout, grads['gamma' + str(i)], grads['beta' + str(i)] = batchnorm_backward(dout, norm_cache)
                        
                  elif self.normalization == 'layernorm':
                    dout, grads['gamma' + str(i)], grads['beta' + str(i)] = layernorm_backward(dout, norm_cache)
                        
                  dout, grads[wi], grads[bi] = affine_backward(dout, fc_cache)

              else:
                dout, grads[wi], grads[bi] = affine_relu_backward(dout, caches[i])
                    
          grads[wi] += self.reg * self.params[wi]
        '''
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
