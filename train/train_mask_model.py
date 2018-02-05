#!/usr/bin/env python

from glob import glob
from PIL import Image
from keras.models import Model
from keras.layers import Input
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
from datetime import datetime
from os import makedirs
import numpy as np
import math
import sys

from tiramisu import Tiramisu
from train_utils import CyclicalCosineWithRestartsLR

def daystamp():
    return datetime.now().strftime('%y%m%d')

def timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')

class image_generator(object):
    def __init__(self, X, Y, batch_size, channels):
        self.X  = X
        self.Y  = Y
        self.bs = batch_size
        self.channels = channels
        self.i  = 0
    def __next__(self):
        xs = self.X[self.i:self.i+self.bs]
        ys = self.Y[self.i:self.i+self.bs]
        # convert 32,64,64,1 -> 32,4096,1
        ys = ys.reshape(len(ys),-1,self.channels)
        self.i = (self.i + self.bs) % self.X.shape[0]
        return xs, ys

def get_score_mask_images(path):
    score_image_names = glob(path+'/diced_score*png')
    mask_image_names = glob(path+'/diced_mask*png')
    score_image_names.sort()
    mask_image_names.sort()
    score_images = np.stack([np.array(Image.open(fn)) for fn in score_image_names])
    mask_images = np.stack([np.array(Image.open(fn))//255 for fn in mask_image_names])
    score_images.shape = score_images.shape + (1,)
    mask_images.shape = mask_images.shape + (1,)
    assert(score_images.shape == mask_images.shape)
    return score_images, mask_images

def main():
    batch_size              = 128 # up from 32, earlier
    num_epochs              = 100
    num_epochs_per_lr_cycle = 10
    train_rate              = 1e-3 # found via FindLR
    num_labels              = 2
    ts = timestamp()
    ds = daystamp()
    print("run: %s"%(ts))
    tsdir = './logs/mask_'+ts
    makedirs(tsdir)
    print("load images")
    train_score_images, train_mask_images = get_score_mask_images('data/train')
    valid_score_images, valid_mask_images = get_score_mask_images('data/valid')
    num_train_images,rows,cols,channels = train_score_images.shape
    num_valid_images,_,_,_ = valid_score_images.shape
    input_shape = (rows,cols,channels)
    train_generator = image_generator(train_score_images, train_mask_images,
                                      batch_size, channels)
    valid_generator = image_generator(valid_score_images, valid_mask_images,
                                      batch_size, channels)
    print("create model")
    # changed to much simpler model thinking that for black & white
    # that would be fine seems to be right.  Also, tried turning on
    # maxpool and was very disappointed by results.
    model = Tiramisu(num_labels,
                     input_shape,
                     nb_layers_per_block=[2,3,4,5,6],#[4,5,7,10,12],
                     initial_filter=24,#48
                     bottleneck_layers=8,#16
                     growth_rate=8,
                     do_td_maxpool=False)
    print(f"saving model summary data/results/mask_model_{ds}_summary.txt")
    with open(f'data/results/mask_model_{ds}_summary.txt','w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print("compile model")
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=RMSprop(train_rate),
                  metrics=["accuracy"])

    # optionally save model
    if True:#False:
        print(f"saving model data/results/mask_model_{ds}.json")
        json_string = model.to_json()
        with open(f"data/results/mask_model_{ds}.json","w") as f:
            f.write(json_string)
        #sys.exit(0)

    # optionally load weights
    if False:
        print(f"loading model data/results/mask_weights_xxx.h5")
        scan_model.load_weights(f'data/results/mask_weights_xxx.h5')

    print("fit model")
    tbcb = TensorBoard(log_dir=tsdir,
                       histogram_freq=0,
                       write_graph=True,
                       write_images=True)
    lrcb = CyclicalCosineWithRestartsLR((num_train_images//batch_size)*num_epochs_per_lr_cycle,
                                        0.0, train_rate)
    model.fit_generator(train_generator,
                        num_train_images//batch_size, num_epochs,
                        verbose=2,
                        validation_data=valid_generator,
                        validation_steps=num_valid_images//batch_size,
                        callbacks=[tbcb,lrcb])
    print(f"batch size = {batch_size}")
    print(f"save weights data/results/mask_weights_{ts}.h5")
    model.save_weights(f'data/results/mask_weights_{ts}.h5')

if __name__ == "__main__":
    main()
