#!/usr/bin/env python

from datetime import datetime
from os import makedirs
from PIL import Image
from glob import glob
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import sys

import keras
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.optimizers import RMSprop
from keras.utils import np_utils

import resnet

# 0=no_note, 1=rest, 2-129=midi_pitch_value
NUM_PITCH_CATEGORIES=128+1+1
# 0=no_length, 1=1/16th, 2=1/8th, 4=1/4 ... 17=whole
NUM_LENGTH_CATEGORIES=16+1
# key inputs are 0-14, where 7 = no sharps/flats = C
NUM_KEY_BITS=4

def daystamp():
    return datetime.now().strftime('%y%m%d')

def timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')

class dual_generator(object):
    """X1 should be list of score images, X2 should be a list of keys, 
    Y1 should be a list of pitch results, Y2 should be a list of length results.  
    All should be the same length."""
    def __init__(self, X1, X2, Y1, Y2, batch_size, channels):
        self.X1 = X1
        self.X2 = X2
        self.Y1 = Y1
        self.Y2 = Y2
        self.bs = batch_size
        self.channels = channels
        self.i  = 0
    def __next__(self):
        xs1 = self.X1[self.i:self.i+self.bs]
        xs2 = self.X2[self.i:self.i+self.bs]
        ys1 = self.Y1[self.i:self.i+self.bs]
        ys2 = self.Y2[self.i:self.i+self.bs]
        self.i = (self.i + self.bs) % self.X1.shape[0]
        return [xs1, xs2], [ys1, ys2]
    
def get_score_indices(input_image_names):
    """Given a list of filenames with the format foo_bar_baz_NNNNN.png
    return a list of Ns"""
    indices = []
    for name in input_image_names:
        index = int(name.split('.')[0].split('_')[-1])
        indices.append(index)
    return indices

def get_score_images_and_indices(path):
    """get the previously-cropped score images.  These are images that are
    cropped snippets centered on staff lines.  The number at the end
    of the filename is used to line up with the score_info.  These
    indices are also returned.

    """
    input_image_names = glob(path+'/crop_score*png')
    input_image_names.sort()
    input_images = np.stack([np.array(Image.open(fn)) for fn in input_image_names])
    input_images.shape = input_images.shape + (1,)
    #num_images,rows,cols,channels = input_images.shape
    input_indices = get_score_indices(input_image_names)
    return input_images, input_indices

def get_score_info(filename):
    """Load the json info from the cropped score_info file.  This is just
    a list of tuples (pitch,length,key)"""
    with open(filename,'r') as f:
        data = json.load(f)
    return data

def get_pitch_results(indices, score_info):
    """Given a list of indices & the score info, respond with a categorical list of pitch results to train to."""
    pitches = np.zeros(len(indices),dtype=np.float32)
    for i,ti in enumerate(indices):
        (pitch,length,key) = score_info[ti]
        pitches[i] = pitch
    pitches = np_utils.to_categorical(pitches,NUM_PITCH_CATEGORIES)
    return pitches

def get_length_results(indices, score_info):
    """Given a list of indices & the score info, respond with a categorical list of length results to train to."""
    lengths = np.zeros(len(indices),dtype=np.float32)
    for i,ti in enumerate(indices):
        (pitch,length,key) = score_info[ti]
        lengths[i] = length
    lengths = np_utils.to_categorical(lengths,NUM_LENGTH_CATEGORIES)
    return lengths

def get_key_inputs(indices, score_info):
    """Given a list of indices & the score info, respond with a list of key inputs to train with."""
    keys = np.zeros((len(indices),NUM_KEY_BITS),dtype=np.float32)
    for i,ti in enumerate(indices):
        (pitch,length,key) = score_info[ti]
        # convert key value into binary-encoded array
        keys[i] = np.array(list(np.binary_repr(key,width=NUM_KEY_BITS)),dtype=np.float32)
    return keys

def step_decay(epoch):
    initial_lrate = 1e-4
    drop = 0.5
    epochs_drop = 10.0
    lrate = (initial_lrate *
             math.pow(drop, math.floor((1+epoch)/epochs_drop)))
    print("step_decay: epoch=%d lrate=%f"%(epoch, lrate))
    return lrate

def main():
    batch_size  = 32
    num_epochs  = 40
    train_rate  = 1e-4
    ts = timestamp()
    ds = daystamp()
    print("run: %s"%(ts))
    print("load images")
    train_score_images, train_score_indices = get_score_images_and_indices('data/train3')
    valid_score_images, valid_score_indices = get_score_images_and_indices('data/valid3')
    num_train_images,rows,cols,channels = train_score_images.shape
    num_valid_images,_,_,_ = valid_score_images.shape
    input_shape = (rows,cols,channels)
    print("load info")
    score_info = get_score_info('data/train3/crop_score_info.json')
    print("score_info len",len(score_info))
    # ----------------------------------------------------------------------
    tsdir = './logs/scan_'+ts
    makedirs(tsdir)
    train_pitch_results  = get_pitch_results(train_score_indices,score_info)
    train_length_results = get_length_results(train_score_indices,score_info)
    train_key_inputs     = get_key_inputs(train_score_indices,score_info)

    valid_pitch_results  = get_pitch_results(valid_score_indices,score_info)
    valid_length_results = get_length_results(valid_score_indices,score_info)
    valid_key_inputs     = get_key_inputs(valid_score_indices,score_info)
    
    train_generator = dual_generator(train_score_images, train_key_inputs,
                                      train_pitch_results, train_length_results,
                                      batch_size, channels)
    valid_generator = dual_generator(valid_score_images, valid_key_inputs,
                                      valid_pitch_results, valid_length_results,
                                      batch_size, channels)
    print("----------------------------------------------------------------------")
    print("create scan model")
    scan_model = resnet.RomerResnetBuilder.build_romer_resnet_18(
        input_shape, NUM_KEY_BITS, NUM_PITCH_CATEGORIES, NUM_LENGTH_CATEGORIES)
    print(f"saving model summary data/results/scan_model_{ds}_summary.txt")
    with open(f'data/results/scan_model_{ds}_summary.txt','w') as f:
        scan_model.summary(print_fn=lambda x: f.write(x + '\n'))
    print("compile scan model")
    scan_model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(train_rate),
                  metrics=["accuracy"])

    # optionally save model
    if True:#False:
        print(f"saving model data/results/scan_model_{ds}.json")
        json_string = scan_model.to_json()
        with open(f"data/results/scan_model_{ds}.json","w") as f:
            f.write(json_string)
        #sys.exit(0)

    print("fit scan model")
    tbcb = TensorBoard(log_dir=tsdir,
                       histogram_freq=0,
                       write_graph=True,
                       write_images=True)
    lrcb = LearningRateScheduler(step_decay)
    scan_model.fit_generator(train_generator,
                             num_train_images//batch_size, num_epochs,
                             verbose=2,
                             validation_data=valid_generator,
                             validation_steps=num_valid_images//batch_size,
                             callbacks=[tbcb,lrcb])
    print(f"batch size = {batch_size}")
    print(f"save weights data/results/scan_weights_{ts}.h5")
    scan_model.save_weights(f'data/results/scan_weights_{ts}.h5')

if __name__ == "__main__":
    main()
