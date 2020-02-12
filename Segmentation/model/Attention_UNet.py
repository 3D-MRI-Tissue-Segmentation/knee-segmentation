from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import concatenate, BatchNormalization, Activation, add, UpSampling2D


def conv2d_bn(input, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(input)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation is None):
        return x

    x = Activation(activation, name=name)(x)

    return x

def conv2d_up(input, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(input)
    x = BatchNormalization(axis=3, scale=False)(x)
    
    return x

def d_conv_block(input, filters, num_row, num_col):

    x = conv2d_bn(input, filters,num_row, num_col)
    #x = conv2d_bn(x, filters,num_row, num_col)

    return x

def u_conv_block(input, filters, num_row, num_col):

    x = conv2d_up(input, filters, num_row, num_col)
    #x = conv2d_up(x, filters,num_row, num_col)

    return x

def Attention_Gate(input_x, input_g, filters, num_row=1, num_col=1, padding='same', strides=(1, 1)):

    x_g = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(input_x)
    x_g = BatchNormalization(axis=-1, scale=False)(x_g)

    x_l = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(input_g)
    x_l = BatchNormalization(axis=-1, scale=False)(x_l)

    x = concatenate([x_g, x_l])
    x = Activation('relu')(x)

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=-1, scale=False)(x)
    alpha = Activation('sigmoid')(x)

    x = alpha*input

    return x


def AttentionUNet(height, width, n_channels):

    input =  Input ((height, width, n_channels))

    ''' number of filters to be applied at different stages 
    of decoding and encoding paths '''
    filters = []

    ''' different possible sizes of kernel size; default is 3,3 
    for square kernels '''
    kernel_size = (3,3)
    n_classes = 2

    #ENCODER PATH

    x1 = d_conv_block(input, filters[0], kernel_size[0], kernel_size[1])
    pool1 = MaxPooling2D(pool_size = (2, 2))(x1)
    
    x2 = d_conv_block(pool1, filters[1], kernel_size[0], kernel_size[1])
    pool2 = MaxPooling2D(pool_size = (2, 2))(x2)

    x3 = d_conv_block(pool2, filters[2], kernel_size[0], kernel_size[1])
    pool3 = MaxPooling2D(pool_size = (2, 2))(x3)

    x4 = d_conv_block(pool3, filters[3], kernel_size[0], kernel_size[1])
    
    #DECODER PATH
    
    a1 = Attention_Gate(x3, x4, filters[2])
    up4 = UpSampling2D(size = (2, 2))(x4)
    y1 = concatenate([up4, a1])
    y1 = u_conv_block(y1, filters[2], kernel_size[0], kernel_size[1])

    a2 = Attention_Gate(x2, y1, filters[1])
    up5 = UpSampling2D(size = (2, 2))(y1)
    y2 = concatenate([up5, a2])
    y2 = u_conv_block(y2, filters[1], kernel_size[0], kernel_size[1])

    a3 = Attention_Gate(x1, y2, filters[0])
    up6 = UpSampling2D(size = (2, 2))(y2)
    y3 = concatenate([up6, a3])
    y3 = u_conv_block(y3, filters[0], kernel_size[0], kernel_size[1])

    output = conv2d_bn (y3, n_classes, 1, 1)

    return output
    


    



