
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
    
    stride = conv_param['stride']
    pad = conv_param['pad']
    N, _, H, W = x.shape
    F, C, HH, WW = w.shape
    H_conv = math.floor(1 + (H + 2 * pad - HH) / stride)
    W_conv = math.floor(1 + (W + 2 * pad - WW) / stride)
    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values=0)
    H += 2 * pad
    W += 2 * pad
    out = np.zeros((N, F, H_conv, W_conv))
    for n in range(N):
        for f in range(F):
            for h in range(H_conv):
                for i in range(W_conv):
                    x_curr = x_pad[n, :, h*stride: h*stride + HH, i*stride: i*stride + WW]
                    w_curr = w[f]
                    out[n, f, h, i] = np.sum(x_curr * w_curr) + b[f]
    
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
    stride = conv_param['stride']
    pad = conv_param['pad']
    N, _, H, W = x.shape
    F, C, HH, WW = w.shape
    H_conv = math.floor(1 + (H + 2 * pad - HH) / stride)
    W_conv = math.floor(1 + (W + 2 * pad - WW) / stride)
    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values=0)
    H += 2 * pad
    W += 2 * pad
    db = np.zeros_like(b)
    dw = np.zeros_like(w)
    for f in range(0, F):
        db[f] = np.sum(dout[:, f, :, :])
    for n in range(N):
        for f in range(F):
            for h in range(H_conv):
                for i in range(W_conv):
                    x_curr = x_pad[n, :, h*stride: h*stride + HH, i*stride: i*stride + WW]
                    dw[f] += dout[n, f, h, i] * x_curr

    dx_pad = np.zeros((N, C, H, W))
    for n in range(N):
        for f in range(F):
            for h in range(H_conv):
                for i in range(W_conv):
                    dx_pad[n, :, h*stride: h*stride + HH, i*stride: i*stride + WW] += dout[n, f, h, i]*w[f]
    dx = dx_pad[:, :, pad:H - pad, pad:W-pad]
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
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_conv = math.floor(1 + (H - pool_height) / stride)
    W_conv = math.floor(1 + (W - pool_width) / stride)
    out = np.zeros((N, C, H_conv, W_conv))
    for n in range(N):
        for c in range(C):
            for h in range(H_conv):
                for w in range(W_conv):
                    out[n, c, h, w] = x[n, c, h*stride: h*stride + pool_height, w*stride: w*stride + pool_width].max()
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
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_conv = math.floor(1 + (H - pool_height) / stride)
    W_conv = math.floor(1 + (W - pool_width) / stride)
    out = np.zeros((N, C, H_conv, W_conv))
    dx = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for h in range(H_conv):
                for w in range(W_conv):
                    max_arg = np.argmax(x[n, c, h*stride: h*stride + pool_height, w*stride: w*stride + pool_width])
                    max_height = math.floor(max_arg / pool_height)
                    max_width = max_arg % pool_height
                    dx[n, c, h*stride+max_height, w*stride+max_width] += dout[n, c, h, w]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx