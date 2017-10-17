# Basing this on the fast.ai Keras Tiramisu implementation of the One
# Hundred Layers Tiramisu as described in Simon Jegou et al.'s paper
# The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for
# Semantic Segmentation.
#
# This is heavily inspired by the code from the 2nd Fast.ai Deep
# Learning course here:
# https://github.com/fastai/courses/blob/master/deeplearning2/tiramisu-keras.ipynb
#
# I've updated it to Keras 2, but intend to rewrite it to make sure I
# understand it all.

from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, concatenate, Conv2DTranspose, Reshape
from keras.regularizers import l2

def relu(x):
    return Activation('relu')(x)

def dropout(x, p):
    return Dropout(p)(x) if p else x

def bn(x):
    return BatchNormalization(axis=-1)(x) # No longer can do mode=2

def relu_bn(x):
    return relu(bn(x))

def conv(x, nf, sz, wd, p, stride=1):
    x = Conv2D(nf, (sz, sz),
               padding="same",
               kernel_initializer="he_uniform",
               strides=(stride,stride),
               kernel_regularizer=l2(wd))(x)
    return dropout(x, p)

def conv_relu_bn(x, nf, sz=3, wd=0, p=0, stride=1):
    return conv(relu_bn(x), nf, sz, wd=wd, p=p, stride=stride)

def dense_block(n,x,growth_rate,p,wd):
    added = []
    for i in range(n):
        b = conv_relu_bn(x, growth_rate, p=p, wd=wd)
        x = concatenate([x, b])
        added.append(b)
    return x,added

def transition_dn(x, p, wd):
#     x = conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd)
#     return MaxPooling2D(strides=(2, 2))(x)
    return conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd, stride=2)

def down_path(x, nb_layers, growth_rate, p, wd):
    skips = []
    for i,n in enumerate(nb_layers):
        x,added = dense_block(n,x,growth_rate,p,wd)
        skips.append(x)
        x = transition_dn(x, p=p, wd=wd)
    return skips, added

def transition_up(added, wd=0):
    x = concatenate(added)
    _,r,c,ch = x.get_shape().as_list()
    return Conv2DTranspose(ch, (3, 3),
                           strides=(2, 2),
                           padding="same",
                           kernel_initializer="he_uniform",
                           kernel_regularizer=l2(wd))(x)
#     x = UpSampling2D()(x)
#     return conv(x, ch, 2, wd, 0)

def up_path(added, skips, nb_layers, growth_rate, p, wd):
    for i,n in enumerate(nb_layers):
        x = transition_up(added, wd)
        x = concatenate([x,skips[i]])
        x,added = dense_block(n,x,growth_rate,p,wd)
    return x

def reverse(a):
    return list(reversed(a))

def create_tiramisu(nb_classes, img_input, nb_dense_block=6,
    growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0):
    """
    nb_classes: number of classes
    img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
    depth: number or layers
    nb_dense_block: number of dense blocks to add to end (generally = 3)
    growth_rate: number of filters to add per dense block
    nb_filter: initial number of filters
    nb_layers_per_block: number of layers in each dense block.
    If positive integer, a set number of layers per dense block.
    If list, nb_layer is used as provided
    p: dropout rate
    wd: weight decay
    """

    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else:
        nb_layers = [nb_layers_per_block] * nb_dense_block

    x = conv(img_input, nb_filter, 3, wd, 0)
    skips,added = down_path(x, nb_layers, growth_rate, p, wd)
    x = up_path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)

    x = conv(x, nb_classes, 1, wd, 0)
    _,r,c,f = x.get_shape().as_list()
    x = Reshape((-1, nb_classes))(x)
    return Activation('softmax')(x)
