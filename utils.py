import tensorflow as tf

def new_weights(shape):
    return tf.get_variable(name='weight', 
                           shape=shape,
                           dtype=tf.float32, 
                           initializer=tf.contrib.layers.xavier_initializer())

def new_biases(length):
    return tf.get_variable(name='bias', 
                           shape=[length],
                           dtype=tf.float32, 
                           initializer=tf.constant_initializer(0.0))

def conv_layer(x, filter_size, num_filters, stride=1, activation=False, bias=True):
    num_input_channels=x.get_shape().as_list()[3]
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)

    layer = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')    
    if bias:
        biases = new_biases(length=num_filters)
        layer = tf.nn.bias_add(layer, biases)
    if activation:
        layer=relu_layer(layer)   
    return layer

def relu_layer(x):
    return tf.nn.relu(x)

def max_pooling_layer(x):
    return tf.nn.max_pool(value=x,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

def resBlock_EDSR(x, filter_size=3, num_filters=64, scale=1):
    with tf.variable_scope('conv1'):
        tmp = conv_layer(x,filter_size,num_filters)
    
    tmp = relu_layer(tmp)
    
    with tf.variable_scope('conv2'):
        tmp = conv_layer(tmp,filter_size,num_filters)
    tmp *= scale
    return x + tmp

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat

def fc_layer(x,         
             num_outputs):
    num_inputs=x.get_shape().as_list()[1]
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(x, weights) + biases
    return layer

def pixel_shuffle_layer(x, r): #n_split=3 for color images, 1 for grayscale
    x = tf.transpose(x,(0,3,1,2))
    
    batch_size, c, h, w = x.get_shape().as_list()
    batch_size = tf.shape(x)[0] #Handling Dimension(None) type for undefined batch dim, uncomment it if necessary
    c //= r ** 2
    out_height = h * r
    out_width = w * r
    
    x=tf.reshape(x, (batch_size,c,r,r,h,w)) 
    shuffle_out = tf.transpose(x,(0, 1, 4, 2, 5, 3))
    shuffle_out=tf.reshape(shuffle_out,(batch_size, c, out_height, out_width))
    
    shuffle_out=tf.transpose(shuffle_out,(0,2,3,1))
    return shuffle_out
    
def upsample_EDSR(x,scale=2,features=64,activation=False,bias=True):
    assert scale in [2,3,4]
    if scale == 2:
        ps_features = features*(scale**2)
        x = conv_layer(x,3,ps_features,1,False,bias)
        x = pixel_shuffle_layer(x,2)
        if activation:
            x = relu_layer(x)            
    elif scale == 3:
        ps_features =features*(scale**2)
        x = conv_layer(x,3,ps_features,1,False,bias)
        x = pixel_shuffle_layer(x,3)
        if activation:
            x = relu_layer(x)    
    elif scale == 4:
        ps_features = features*(2**2)
        for i in range(2):
            x = conv_layer(x,3,ps_features,1,False,bias)
            x = pixel_shuffle_layer(x,2)
            if activation:
                x = relu_layer(x)    
    return x

def tf_warp3(img, flows, H, W):
    with tf.variable_scope('warping_function'):
        shape = img.get_shape().as_list()
        num_batch = tf.shape(img)[0]
        channels = shape[3]
        
        x_t = tf.matmul(tf.ones(shape=tf.stack([H, 1])),
                                tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, W), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, H), 1),
                        tf.ones(shape=tf.stack([1, W])))
    
        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))
        grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat])
        grid = tf.expand_dims(grid,0)
        grid = tf.reshape(grid, [-1])
        grid = tf.tile(grid, tf.stack([num_batch]))
        grid = tf.reshape(grid, tf.stack([num_batch, 2, H*W]))
            
        flows_flat = tf.reshape(flows, [num_batch, grid.get_shape().as_list()[1], grid.get_shape().as_list()[2]])
        flows_t = flows_flat + grid
    
        x_s = tf.slice(flows_t, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(flows_t, [0, 1, 0], [-1, 1, -1])
        x = tf.reshape(x_s, [-1])
        y = tf.reshape(y_s, [-1])
              
        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        H_f = tf.cast(H, 'float32')
        W_f = tf.cast(W, 'float32')
        zero = tf.zeros([], dtype='int32')
        max_y = tf.cast(H - 1, 'int32')
        max_x = tf.cast(W - 1, 'int32')
    
        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0)*(W_f - 1) / 2.0
        y = (y + 1.0)*(H_f - 1) / 2.0
    
        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1
    
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        
        dim2 = W
        dim1 = W*H
        base = repeat(tf.range(num_batch)*dim1, H*W)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1
    
        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(img, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)
    
        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
        wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
        wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
        wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
        output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
        output = tf.reshape(output, tf.stack([num_batch, H, W, channels]))
    return output

def repeat(x, n_repeats):
    with tf.variable_scope('repeat'):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])
    
