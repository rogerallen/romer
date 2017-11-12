# -*- coding: utf-8 -*-
'''One Hundred Layers Tiramisu model for Keras

Model as described in "The One Hundred Layers Tiramisu: Fully
Convolutional DenseNets for Semantic Segmentation" by Simon Jégou,
Michal Drozdzal, David Vazquez, Adriana Romero, Yoshua
Bengio. https://arxiv.org/abs/1611.09326

Originally based on the code from the 2nd Fast.ai Deep Learning course
at
https://github.com/fastai/courses/blob/master/deeplearning2/tiramisu-keras.ipynb
which is Apache licensed. See
https://github.com/fastai/courses/blob/master/LICENSE.txt

I updated the code to Keras 2 and rewrote it based on my own tastes.
Since IANAL, let's keep the Apache license for this code, too.

Copyright (c) 2017 Roger Allen

'''

from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Reshape
from keras.layers import concatenate
from keras.models import Model
from keras.regularizers import l2

def __dense_block_down(x, nb_layers, growth_rate, p, wd, name):
    """Described in Figure 1 & 2 from the paper.  A dense_block has
    nb_layers of 'Layer' blocks.

    Each Layer block is a described in Table 1 of the paper.  Batch
    Norm, ReLU, 3x3 Conv of 'growth_rate' filters, Dropout

    Note in Figure 1 where the down pathway has a post-DB concat of
    the x input and the up connection does not.  This 'down'
    dense_block returns a concatenation of x and the intermediate
    layer outputs: [x,a,a',a'',...].

    """
    for i in range(nb_layers):
        a = BatchNormalization(axis=-1,name=name+f'_{i}_batchnorm')(x)
        a = Activation('relu',name=name+f'_{i}_relu')(a)
        a = Conv2D(growth_rate, (3, 3),
                   padding="same",
                   kernel_initializer="he_uniform",
                   kernel_regularizer=l2(wd),
                   name=name+f'_{i}_conv2d')(a)
        a = Dropout(p, name=name+f'_{i}_dropout')(a) if p else a
        x = concatenate([x, a],name=name+f'_{i}_concat')
    return x

def __dense_block_up(x, nb_layers, growth_rate, p, wd, name):
    """Described in Figure 1 & 2 from the paper.  A dense_block has
    nb_layers of 'Layer' blocks.

    Each Layer block is a described in Table 1 of the paper.  Batch
    Norm, ReLU, 3x3 Conv of 'growth_rate' filters, Dropout

    Note in Figure 1 where the down pathway has a post-DB concat of
    the x input and the up connection does not.  This 'up' dense_block
    returns a concatenated layer of only the added layer outputs
    [a,a',a'',...], not including the original input.

    """
    for i in range(nb_layers):
        a = BatchNormalization(axis=-1, name=name+f'_{i}_batchnorm')(x)
        a = Activation('relu', name=name+f'_{i}_relu')(a)
        a = Conv2D(growth_rate, (3, 3),
                   padding="same",
                   kernel_initializer="he_uniform",
                   kernel_regularizer=l2(wd),
                   name=name+f'_{i}_conv2d')(a)
        a = Dropout(p, name=name+f'_{i}_dropout')(a) if p else a
        x = concatenate([x, a], name=name+f'_{i}_concat_xa')
        if i == 0:
            added = a
        else:
            added = concatenate([added,a], name=name+f'_{i}_concat_aa')
    return added

def __transition_down(x, p, wd, do_maxpool, name):
    """See Table 1 for the Transition Down block.  Similar to the Layer in
    the DenseBlock, but slightly different with 1x1 Conv and 2x2
    MaxPooling.  Jeremy Howard's fast.ai code did not use MaxPooling,
    but rather just 2x strides, so allow for control of this via
    do_maxpool.

    """
    nb_filter = x.get_shape().as_list()[-1]
    conv_stride = 1 if do_maxpool else 2
    x = BatchNormalization(axis=-1, name=name+'_batchnorm')(x)
    x = Activation('relu', name=name+'_relu')(x)
    x = Conv2D(nb_filter, (1, 1),
               padding="same",
               kernel_initializer="he_uniform",
               strides=(conv_stride, conv_stride),
               kernel_regularizer=l2(wd),
               name=name+'_conv2d')(x)
    x = Dropout(p, name=name+'_dropout')(x) if p else x
    x = MaxPooling2D(name=name+'_maxpool')(x) if do_maxpool else x
    return x

def __transition_up(added, wd=0, name="TU"):
    """See Table 1 for the Transition Up block. A 3x3 Transposed Conv with
    stride=2.

    """
    _,r,c,ch = added.get_shape().as_list()
    x = Conv2DTranspose(ch, (3, 3),
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer="he_uniform",
                        kernel_regularizer=l2(wd),
                        name=name+'_conv2d')(added)
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
    """Instantiate the tiramisu architecture.  See Figures 1, 2 and Table
    1 for diagrams showing what each of these settings are.

    inputs:
      nb_classes: number of output classes to classify.

      input_shape: tuple of shape (channels, rows, columns) if
          K.image_data_format() == 'channels_first', else (rows,
          columns, channels).

      nb_layers_per_block: list describing down and up pathway. Items
          in list are number of layers in each dense block (not
          including bottleneck) on the way down and the reverse of
          this on the way back up.  Set to [4,5,7,10,12] to match the
          paper.

      initial_filter: number of filters in initial 3x3 Conv
          (48 per paper)

      bottleneck_layers: number of layers in bottleneck stage between
          the down and up paths.  (15 per paper)

      growth_rate: number of filters to add per dense block
          (12 or 16 per paper)

      do_td_maxpool: Jeremy Howard's implemntation allowed for
          removing the TD block's maxpooling layer.  Default set to
          'True' to match the paper, but my experience matches his in
          that setting this 'False' results in better results.

      p: dropout rate or None for no Dropout() (0.2 per paper)

      wd: weight decay (1e-4 per paper)

    returns:
      Tiramisu Keras Model

    """

    assert type(nb_layers_per_block) == list

    image_input = Input(shape=input_shape,name="image_input")

    # initial 3x3 Convolution
    x = Conv2D(initial_filter, (3, 3), padding="same",
               kernel_initializer="he_uniform",
               kernel_regularizer=l2(wd),
               name="initial_3x3_conv2d")(image_input)

    # Down Path: DenseBlocks + TransitionDowns
    skips = []
    for i,nb_layers in enumerate(nb_layers_per_block):
        #print(f"down {i} {nb_layers}")
        x = __dense_block_down(x, nb_layers, growth_rate, p, wd, f'DB_DN{i}')
        skips.append(x)
        x = __transition_down(x, p, wd, do_td_maxpool, f'TD{i}')

    # Bottleneck
    added = __dense_block_up(x, bottleneck_layers, growth_rate, p, wd, 'DB_bottleneck')

    # Up Path: TransitionUp + DenseBlocks
    skips = list(reversed(skips))
    nb_layers_per_block = list(reversed(nb_layers_per_block))
    for i,nb_layers in enumerate(nb_layers_per_block):
        ir = len(nb_layers_per_block)-1-i
        #print(f"up {ir} {nb_layers}")
        x = __transition_up(added, wd, f'TU{ir}')
        x = concatenate([x,skips[i]], name=f"skip{ir}_concat")
        added = __dense_block_up(x, nb_layers, growth_rate, p, wd, f'DB_UP{ir}')

    # final 1x1 Convolution & Softmax
    x = Conv2D(nb_classes, (1, 1), padding="same",
               kernel_initializer="he_uniform",
               kernel_regularizer=l2(wd),
               name="final_conv2d")(added)
    x = Reshape((-1, nb_classes), name="final_reshape")(x)
    x = Activation('softmax', name="final_softmax")(x)

    model = Model(image_input, x, name='tiramisu')

    return model
