# -*- coding: utf-8 -*-
'''One Hundred Layers Tiramisu model for Keras

Model as described in "The One Hundred Layers Tiramisu: Fully
Convolutional DenseNets for Semantic Segmentation" by Simon Jégou,
Michal Drozdzal, David Vazquez, Adriana Romero, Yoshua
Bengio. https://arxiv.org/abs/1611.09326

Originally based on the code from the 2nd Fast.ai Deep Learning course
here:
https://github.com/fastai/courses/blob/master/deeplearning2/tiramisu-keras.ipynb

Updated to Keras 2 and rewritten based on my own sensibilities.

MIT License

Copyright (c) 2017 Roger Allen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import concatenate
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K

def __dense_block_down(x, nb_layers, growth_rate, p, wd):
    """Described in Figure 1 & 2 from the paper.  A dense_block has
    nb_layers of 'Layer' blocks.

    Each Layer block is a described in Table 1 of the paper.  Batch
    Norm, ReLU, 3x3 Conv of 'growth_rate' filters, Dropout

    Note in Figure 1 where the down pathway has a post-DB concat of
    the x input and the up connection does not.  This 'down'
    dense_block returns a concatenation of x and the intermediate
    layer outputs: [x,a,a',a'',a''',...].

    """
    with K.name_scope('DenseBlockDown'):
        for i in range(nb_layers):
            a = BatchNormalization(axis=-1)(x)
            a = Activation('relu')(a)
            a = Conv2D(growth_rate, (3, 3),
                       padding="same",
                       kernel_initializer="he_uniform",
                       kernel_regularizer=l2(wd))(a)
            a = Dropout(p)(a) if p else a
            x = concatenate([x, a])
    return x

def __dense_block_up(x, nb_layers, growth_rate, p, wd):
    """Described in Figure 1 & 2 from the paper.  A dense_block has
    nb_layers of 'Layer' blocks.

    Each Layer block is a described in Table 1 of the paper.  Batch
    Norm, ReLU, 3x3 Conv of 'growth_rate' filters, Dropout

    Note in Figure 1 where the down pathway has a post-DB concat of
    the x input and the up connection does not.  This 'up' dense_block
    returns a concatenated layer of only the added layer outputs
    [a,a',a'',a''',...], not including the original input.

    """
    with K.name_scope('DenseBlockUp'):
        for i in range(nb_layers):
            a = BatchNormalization(axis=-1)(x)
            a = Activation('relu')(a)
            a = Conv2D(growth_rate, (3, 3),
                       padding="same",
                       kernel_initializer="he_uniform",
                       kernel_regularizer=l2(wd))(a)
            a = Dropout(p)(a) if p else a
            x = concatenate([x, a])
            if i == 0:
                added = a
            else:
                added = concatenate([added,a])
    return added

def __transition_down(x, p, wd, do_maxpool):
    """See Table 1 for the Transition Down block.  Similar to the Layer in
    the DenseBlock, but slightly different with 1x1 Conv and 2x2
    MaxPooling.  Jeremy Howard's fast.ai code did not use MaxPooling,
    but rather just 2x strides, so allow for control of this via
    do_maxpool.

    """
    with K.name_scope('TransitionDown'):
        nb_filter = x.get_shape().as_list()[-1]
        conv_stride = 1 if do_maxpool else 2
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = Conv2D(nb_filter, (1, 1),
                   padding="same",
                   kernel_initializer="he_uniform",
                   strides=(conv_stride, conv_stride),
                   kernel_regularizer=l2(wd))(x)
        x = Dropout(p)(x) if p else x
        x = MaxPooling2D()(x) if do_maxpool else x
    return x

def __transition_up(added, wd=0):
    """See Table 1 for the Transition Up block. A 3x3 Transposed Conv with
    stride=2.

    """
    with K.name_scope('TransitionUp'):
        _,r,c,ch = added.get_shape().as_list()
        x = Conv2DTranspose(ch, (3, 3),
                            strides=(2, 2),
                            padding="same",
                            kernel_initializer="he_uniform",
                            kernel_regularizer=l2(wd))(added)
    return x

def Tiramisu(nb_classes,
             input_shape,
             nb_layers_per_block,
             initial_filter=48,
             bottleneck_layers=15,
             growth_rate=16,
             do_td_maxpool=True,
             p=0.2,
             wd=1e-4):
    """Instantiate the tiramisu architecture.

    inputs:
      nb_classes: number of output classes to classify

      input_shape: tuple of shape (channels, rows, columns) or (rows,
          columns, channels) that should match your
          K.image_data_format() == 'channels_first' setting.

      nb_layers_per_block: items in list are number of layers in each
          dense block (not including bottleneck) e.g. [4,5,7,10,12] to
          match the paper.  Note that the last entry in the list
          doesn't produce skips and the up path won't mirror it.

      initial_filter: number of filters in initial 3x3 Conv
          (48 per paper)

      bottleneck_layers: number of layers in bottleneck stage
          (15 per paper)

      growth_rate: number of filters to add per dense block
          (12 or 16 per paper)

      do_td_maxpool: Jeremy Howard's implemntation allowed for
          removing the TD block's maxpooling layer.

      p: dropout rate or None for no Dropout() (0.2 per paper)

      wd: weight decay (1e-4 per paper)

    returns:
      Tiramisu Keras Model

    """

    assert type(nb_layers_per_block) == list

    image_input = Input(shape=input_shape)

    # initial 3x3 Convolution
    x = Conv2D(initial_filter, (3, 3), padding="same",
               kernel_initializer="he_uniform",
               kernel_regularizer=l2(wd))(image_input)

    # Down Path: DenseBlocks + TransitionDowns
    skips = []
    for nb_layers in nb_layers_per_block:
        x = __dense_block_down(x, nb_layers, growth_rate, p, wd)
        skips.append(x)
        x = __transition_down(x, p, wd, do_td_maxpool)

    # Bottleneck
    added = __dense_block_up(x, bottleneck_layers, growth_rate, p, wd)

    # Up Path: TransitionUp + DenseBlocks
    skips = list(reversed(skips))
    nb_layers_per_block = list(reversed(nb_layers_per_block))
    for i,nb_layers in enumerate(nb_layers_per_block):
        x = __transition_up(added, wd)
        x = concatenate([x,skips[i]])
        added = __dense_block_up(x, nb_layers, growth_rate, p, wd)

    # final 1x1 Convolution & Softmax
    x = Conv2D(nb_classes, (1, 1), padding="same",
               kernel_initializer="he_uniform",
               kernel_regularizer=l2(wd))(x)
    x = Reshape((-1, nb_classes))(x)
    x = Activation('softmax')(x)

    model = Model(image_input, x, name='tiramisu')

    return model
