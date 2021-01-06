# -*- coding: utf-8 -*-
from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # reshape(-1)의 의미
    # https://yganalyst.github.io/data_handling/memo_5/
    
    #out = XdotW + b
    out = x.reshape(x.shape[0], -1).dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = np.dot(dout, w.T) #크기 = (N, M) * (M, D) = (N, D)
    dx = np.reshape(dx, x.shape) #크기 = (N, d_1, ... d_k)
    x_row = x.reshape(x.shape[0], -1) #크기 = (N, D)
    dw = np.dot(x_row.T, dout) #크기 = (D, N) * (N, M) = (D, M)
    db = np.sum(dout, axis = 0) #크기 = (M, )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout
    dx[x < 0] = 0 #relu의 특성에 맞게 0보다 작으면 다 0으로.
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        # 참고: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

        mean = np.mean(x, axis = 0)
        var = np.var(x, axis= 0) 

        input_minus_mean = x- mean
        sqrt_variance_plus_eps = np.sqrt(var + eps)
        invert_sqrt_variance_plus_eps = 1. / sqrt_variance_plus_eps

        x_normalization = input_minus_mean * invert_sqrt_variance_plus_eps # (N, D)
        out = gamma * x_normalization + beta
                
        running_mean = momentum * running_mean + (1-momentum) * mean
        running_var = momentum * running_var + (1-momentum) * var

        cache = (x_normalization, gamma, input_minus_mean, invert_sqrt_variance_plus_eps, sqrt_variance_plus_eps, var, eps)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_normalization = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_normalization + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    #https://kimcando94.tistory.com/110
    x_normalization, gamma, input_minus_mean, invert_sqrt_variance_plus_eps, sqrt_variance_plus_eps, var, eps = cache
    N, D = dout.shape

    #step 9
    dbeta = np.sum(dout, axis = 0, keepdims = True)

    #step 8
    dgamma = np.sum(dout * x_normalization, axis = 0, keepdims = True)
    dx_normalization = dout * gamma # (N,D)

    #step 7
    dinvert_sqrt_variance_plus_eps = np.sum(dx_normalization * input_minus_mean, axis = 0)
    dinput_minus_mean = dx_normalization * invert_sqrt_variance_plus_eps
    
    #step 6
    dsqrt_variance_plus_eps = (-1. / (sqrt_variance_plus_eps ** 2)) * dinvert_sqrt_variance_plus_eps
    
    #step 5
    dvar = 0.5 * (1./ np.sqrt(var+eps)) * dsqrt_variance_plus_eps 

    #step 4
    dsq = (1./ N) * np.ones((N, D)) * dvar

    #step 3
    dinput_minus_mean_2 = 2 * input_minus_mean * dsq

    #step 2
    dx1 = dinput_minus_mean + dinput_minus_mean_2
    dmean = -1 * np.sum(dx1, axis = 0)

    #step 1
    dx2 = (1. / N) * np.ones((N, D)) * dmean

    #step 0
    dx = dx1 + dx2

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.
    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.
    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x_normalization, gamma, input_minus_mean, invert_sqrt_variance_plus_eps, sqrt_variance_plus_eps, var, eps = cache
    N, D = dout.shape
    
    dx_normalization = dout * gamma
    
    dx = (1. / N) * invert_sqrt_variance_plus_eps * (N * dx_normalization - np.sum(dx_normalization, axis=0) - x_normalization * np.sum(dx_normalization * x_normalization, axis=0))
    
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalization, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x*mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    #batch-size, channel(image의 채널 수), (input image의) height, width
    B, C, H, W = x.shape

    #learnable filter의 shape
    #filter의 개수, channel 개수, filter image의 height, weight 
    F_N, _, F_H, F_W = w.shape
    stride, padding = conv_param['stride'], conv_param['pad']

    #Convnet의 output인 activation map의 size를 구하는 공식
    #http://taewan.kim/post/cnn/
    #https://stackoverflow.com/questions/28232235/how-to-calculate-the-number-of-parameters-of-convolutional-neural-networks?fbclid=IwAR0wIAcfo6ZWnUOm6JpWOnQrgvJpCtZ9-5u9yd5_CUkCbtu00eqhnr6VCI8
    
    output_height = (H + 2 * padding - F_H) // stride + 1
    output_width = (W + 2 * padding - F_W) // stride + 1

    out = np.zeros((B, F_N, output_height, output_width))

    #이미지에 해당하는 영역에만 padding을 적용한다
    x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values = 0)
    
    for n in range(B):       # nth image
      for f in range(F_N):   # fth filter
        for h_i in range(output_height):
          for w_i in range(output_width):
            #각각의 filter를 적용시켜서 convolution의 결과값인 스칼라값 하나를 activation map에 대입하기
            out[n, f, h_i, w_i] = np.sum(
            x_pad[n, 
              #3 channel을 모두 구한 다음 np.sum을 해주기 위해서 channel을 전체 선택한다
                  :, 
              # filter size만큼 input image에서 가져온다
              # filter size인 W의 특정 filter를 cropped input에 곱함으로, convolution 연산을 진행한다.
              # element wise multiplication -> 행렬의 "각각의 요소"끼리 곱해준다.
              # 곱셈이 끝난 다음에, 전체 값을 더해준다(np.sum), 그리고 bias를 더한다.
                  h_i*stride : h_i*stride+F_H,
                  w_i*stride : w_i*stride + F_W] * w[f]) + b[f]
              
    #결국 이것은 cropped input에 대해서 Wx(각각의 요소를 곱해주는 것)+b를 하는 것과 동일하다.
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    B, C, H, W = x.shape
    F_N, _, F_H, F_W = w.shape
    stride, padding = conv_param['stride'], conv_param['pad']
    _, _, output_height, output_width = dout.shape
    
    x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    for n in range(B):  # nth image
      for f in range(F_N): # fth filter
        db[f] += np.sum(dout[n, f])
        for h_i in range(output_height):
          for w_i in range(output_width):
            # 각 filter를 돌면서 backpropagation을 진행한다.
            # 요소 곱 - Wx + b가 진행되어 있는 상태에서,
            # 최종 gradient는 각각의 element가 사용된 수식들에서 각각 구한 gradient들의 합이다.
            # batch normalization의 backpropagation을 생각해 보자.
            # ex) x-> Wx +b -> output_a -> output_a + output+b -> output_c -> Zx -> output_b
            # 이러한 네트워크에서 x에 대한 gradient는 output_c부터 시작하여 내려오게 되는데
            # Wx+b와 ZW는 chain rule에 의해서 독립적으로 gradient를 구할 수 있다 
            # 따라서, x의 gradient = dx는 Wx+b로부터 구한 dx_1와 Zx로부터 구한 dx_2의 합이다.
            # 그래서 dx = dx_1 + dx_2
            # 이를 convolution network에 적용시켜 보자. 
            
            # w에 대한 gradient을 구해보자.
            # forward에서 진행한 요소 곱 Wx+b를 W에 대하여 편미분해보자.
            # 그러면 X만 남게 되는데, 이는 forward에서 crop된 특정 영역의 input 이다.
            # (이때 shape == filter shape)
            # chain rule을 적용하기 위하여 dout를 구해주면, dW의 일부분을 구하게 되는데,
            # 이는 forward에서 filter가 슬라이딩하면서 건드렸던 모든 input에 대한 dW를 의미한다.
            # 우리가 구해야 하는 dW는 W가 연산에 활용되었던 X의 모든 구간에 대한 gradient를 구해야 하니까,
            # += 연산을 통하여 각각의 구간들에 대한 gradient를 합해야 한다.

            dw[f] += x_pad[n, :,
                      h_i*stride : h_i*stride + F_H, 
                      w_i*stride : w_i*stride + F_W] * dout[n, f, h_i, w_i]                
                    
            # x에 대한 gradient를 구해보자.
            # w와 마찬가지로 연산을 진행하였던 모든 x에 대한 gradient를 구한 다음 합해야 한다.
            dx_pad[n, :, 
                  h_i*stride : h_i*stride + F_H, 
                  w_i*stride : w_i*stride + F_W] += (w[f] * dout[n, f, h_i, w_i])
                    
    # Unpad
    dx = dx_pad[:, :, padding:padding+H, padding:padding+W]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    B, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H_max = (H - pool_height) // stride + 1
    W_max = (W - pool_width) // stride + 1
    
    out = np.zeros((B, C, H_max, W_max))
    
    for n in range(B):
        for h_i in range(H_max):
          for w_i in range(W_max):
            #axis를 image의 height x width 영역에 잘 맞춰주기
              out[n, :, h_i, w_i] = np.max(x[n, :, 
                  h_i*stride : h_i*stride + pool_height,
                  w_i*stride : w_i*stride + pool_width], axis = (-1, -2))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    B, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H_max = (H - pool_height) // stride + 1
    W_max = (W - pool_width) // stride + 1

    dx = np.zeros_like(x)
    
    for n in range(B):
      for f in range(C):
        for h_i in range(H_max):
            for w_i in range(W_max):
              #max pool의 backpropagation은 relu의 backprop와 비슷하다.
              #(max pooling 특성상) max 값으로 적용된 data만 pooling layer에서 연산이 적용되고, 나머지는 영향을 끼치지 않기 때문에
              #forward 단계에서 max로 적용되었된 data 부분만 dout를 흘려주고, 나머지는 0을 넣어준다.
              

              #np.unravel_index 설명
              #https://qastack.kr/programming/5469286/how-to-get-the-index-of-a-maximum-element-in-a-numpy-array-along-one-axis
              index = np.unravel_index(np.argmax(x[n, f, 
                      h_i*stride : h_i*stride + pool_height,
                      w_i*stride : w_i*stride + pool_width]), (pool_height, pool_width))

              dx[n, f, h_i*stride : h_i*stride + pool_height,
                      w_i*stride : w_i*stride + pool_width][index] = dout[n, f, h_i, w_i]


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = x.shape
    x_new = x.transpose(0, 2, 3, 1).reshape(N*H*W, C)
    out, cache = batchnorm_forward(x_new, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    #flatten 했던 걸 다시... 확장시키는? 그런 것인 듯?
    N, C, H, W = dout.shape
    dout_new = dout.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    dx, dgamma, dbeta = batchnorm_backward(dout_new, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
