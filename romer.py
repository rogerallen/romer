#!/usr/bin/env python3
"""romer.py - read sheet music

Copyright (c) 2018, Roger Allen

This project is licensed under the GPL V3 License - see
https://www.gnu.org/licenses/gpl-3.0.en.html for details

"""
import os
import sys
import logging
import argparse
import configparser

import numpy as np
from PIL import Image
from skimage import measure
import json

from keras.models import model_from_json

NUM_KEY_BITS=4

# ======================================================================
def get_input_prediction_tiles(input_image):
    """Dice up the image to predict.  We have to send 64x64 tiles"""
    num_tiles = np.ceil(np.array([input_image.width/64,input_image.height/64]))
    tot_tiles = int(num_tiles[0]*num_tiles[1])
    input_tiles = np.zeros((tot_tiles,64,64,1),dtype='uint8')
    i = 0
    for iy in range(int(num_tiles[1])):
        for ix in range(int(num_tiles[0])):
            input_tiles[i] = np.array(input_image.crop([ix*64,iy*64,(ix+1)*64,(iy+1)*64])).reshape(64,64,1)
            i += 1
    return input_tiles

def image_from_tiles(width,height,num_input_tiles,pred_tiles,pred_channel):
    # reshape & collect a single channel plane to visualize
    pred_tiles = pred_tiles.reshape(num_input_tiles,64,64,2)
    pred_one_tiles = pred_tiles[:,:,:,1]
    num_tiles = np.ceil(np.array([width/64,height/64]))
    pred_one = Image.new('L', (int(num_tiles[0])*64,int(num_tiles[1])*64), (0,))
    i = 0
    for iy in range(int(num_tiles[1])):
        for ix in range(int(num_tiles[0])):
            pred_tile_image = Image.fromarray(np.uint8(pred_one_tiles[i]*255))
            pred_one.paste(pred_tile_image, (ix*64,iy*64))
            i += 1
    return pred_one

# make greyscale image binary
def my_threshold(x):
    return 255 if x > 250 else 0

my_threshold = np.vectorize(my_threshold)

def get_mask_labels(mask_image):
    mask_array = np.reshape(np.asarray(mask_image),
                            (mask_image.height, mask_image.width))
    mask_array = my_threshold(mask_array)
    # label individual regions via skimage.measure.label
    mask_labels, num_labels = measure.label(mask_array,
                                            return_num=True,
                                            background=0.0,
                                            connectivity=2)
    return mask_labels, num_labels

def get_good_mask_rects(mask_labels, num_labels, threshold_size):
    # find rectangles and return as array of tuples (x0,y0,w,h)
    regions = []
    mask_props = measure.regionprops(mask_labels)
    ave_h = 0
    num_good_labels = 0
    for i in range(num_labels):
        if mask_props[i].area > threshold_size:
            (y0,x0,y1,x1) = mask_props[i].bbox
            h = (y1-y0)
            ave_h += h
            num_good_labels += 1
    ave_h /= num_good_labels
    ave_h2 = ave_h/2
    logging.debug("good rects = %d"%(num_good_labels))
    for i in range(num_labels):
        if mask_props[i].area > threshold_size:
            (y0,x0,y1,x1) = mask_props[i].bbox
            yc,xc = mask_props[i].centroid
            h = (y1-y0)
            h2 = h/2
            # m = 0 FIXME to deal with rotations
            regions.append((x0,int(yc-ave_h2),(x1-x0),int(ave_h),0.0,int(yc)))
    if len(regions) == 0:
        logging.warning("No good rects found")
        return []
    # FIXME -- this pass probably isn't needed?
    # 120 is the height of the staff scanning network input
    out_h2 = 120/2 # need them all to have the same size
    rects = []
    for i in range(len(regions)):
        (x,y,w,h,m,b) = regions[i]
        rects.append((x,y+h//2-out_h2,w,2*out_h2))
    return rects

def save_staff_images(temp_dir, score_image, rects, staff_info):
    for i,(x,y,w,h) in enumerate(rects):
        name = os.path.join(temp_dir,'staff_%02d.png'%(i))
        staff_info.append({"name": name, "x": x, "y": y, "width": w, "height": h})
        score_image.crop((x,y,x+w,y+h)).save(name)

def get_crop_inputs(cur_staff_info):
    img_crop_width = 32
    img_step = 2
    img_crop_height = int(cur_staff_info['height'])
    img_count = cur_staff_info['width']//img_step
    cur_img = Image.open(cur_staff_info['name'])
    cur_crop_inputs = np.zeros((img_count,img_crop_height,img_crop_width,1),
                               dtype='uint8')
    for i in range(img_count):
        img_x = i*img_step
        cur_crop_img = cur_img.crop((img_x, 0, img_x + img_crop_width, img_crop_height))
        cur_crop_inputs[i] = np.array(cur_crop_img).reshape(img_crop_height,img_crop_width,1)
    return cur_crop_inputs

def get_notes_lengths(note_pred,length_pred):
    assert(note_pred.shape[0] == length_pred.shape[0])
    num_preds = note_pred.shape[0]
    notes_lengths = []
    for i in range(num_preds):
        note_class = note_pred[i].argmax()
        length_class = length_pred[i].argmax()
        note_confidence = note_pred[i,note_class]
        #if note_confidence > 0.99:
        #    note_confidence = "high"
        #elif note_confidence > 0.8:
        #    note_confidence = "med"
        #else:
        #    note_confidence = "low"
        note_index = note_class-2
        note_name = "C C# D D# E F F# G G# A A# B".split()[note_index % 12]
        note_octave = int(note_index/12) - 2
        if note_index == -2:
            note_name = 'X'
            note_octave = 0
        elif note_index == -1:
            note_name = 'R'
            note_octave = 0
        length_confidence = length_pred[i,length_class]
        #if length_confidence > 0.99:
        #    length_confidence = "high"
        #elif length_confidence > 0.8:
        #    length_confidence = "med"
        #else:
        #    length_confidence = "low"
        length = length_class/4
        if note_index == -2 and note_confidence > 0.8:#note_confidence != "low":
            continue # no note
        notes_lengths.append([i,
                              (note_index,note_name,note_octave,note_confidence),
                              (length, length_confidence)])
    return notes_lengths

def find_best_sample(cur):
    num_samples = len(cur)
    if num_samples > 0:
        conf = np.zeros(num_samples)
        for i,c in enumerate(cur):
            conf[i] = c[0][-1]
        mci = conf.argmax()
        return (cur[mci][0][0],cur[mci][1][0])
    return (None, None)

def refine_notes_lengths(raw_notes_lengths):
    notes_lengths = []
    cur = []
    last_index = -100
    for x in raw_notes_lengths:
        if x[0] == last_index + 1:
            cur.append(x[1:])
        else:
            (n,l) = find_best_sample(cur)
            if n:
                notes_lengths.append((n,l))
            cur = []
        last_index = x[0]
    (n, l) = find_best_sample(cur)
    if n:
        notes_lengths.append((n,l))
    return notes_lengths

def output_rmf(outfile, notes_lengths):
    logging.info("writing output to: %s"%(outfile))
    with open(outfile,'w') as f:
        cur_beat = 0.0
        for (n,l) in notes_lengths:
            print("note,%d,%f,%f"%(n,cur_beat,l),file=f)
            cur_beat += l

# ======================================================================
class Application(object):
    def __init__(self,argv):
        self.config = Config()
        self.parse_args(argv)
        self.adjust_logging_level()

    def run(self):
        """Decode the image to music."""
        logging.info("Args: {}".format(self.args))
        # first, get a mask image to show where the
        # staff lines are
        self.init_mask_model(self.args.model_dir)
        if not os.path.isdir(self.args.temp_dir):
            os.mkdir(self.args.temp_dir)
        mask_image = self.run_mask_model(self.args.input_filename,
                                         self.args.temp_dir)
        # next make those masks into rectangles to snip
        staff_info = self.get_staff_info(mask_image, self.args.temp_dir)
        # now we use models that decode each staff to find notes
        self.init_scan_model(self.args.model_dir)
        # now we scan the staff images and output music_info
        music_info = self.scan_staff_info(staff_info, int(self.args.key))
        # finally we output the music_info
        self.output_music(music_info, self.args.output_filename)
        return 0

    def init_mask_model(self,model_dir):
        logging.info("init_mask_model %s",model_dir)
        model_file   = os.path.join(model_dir,'mask_model.json')
        weights_file = os.path.join(model_dir,'mask_weights.h5')
        with open(model_file,"r") as f:
            json_string = f.read()
        self.mask_model = model_from_json(json_string)
        self.mask_model.load_weights(weights_file)

    def init_scan_model(self,model_dir):
        model_file   = os.path.join(model_dir,'scan_model.json')
        weights_file = os.path.join(model_dir,'scan_weights.h5')
        with open(model_file,"r") as f:
            json_string = f.read()
        self.scan_model = model_from_json(json_string)
        self.scan_model.load_weights(weights_file)

    def run_mask_model(self,input_filename,temp_dir):
        logging.info("run_mask_model %s, %s",input_filename,temp_dir)
        self.score_image = Image.open(input_filename)
        if self.score_image.mode == 'RGBA':
            background = Image.new('RGBA', self.score_image.size, (255,255,255))
            self.score_image = Image.alpha_composite(background, self.score_image)
        if self.score_image != 'L':
            self.score_image = self.score_image.convert('L')
        score_tiles = get_input_prediction_tiles(self.score_image)
        mask_tiles = self.mask_model.predict(score_tiles)
        mask_image = image_from_tiles(self.score_image.width,
                                      self.score_image.height,
                                      score_tiles.shape[0], mask_tiles, 1)
        mask_filename = os.path.join(temp_dir,"mask_image.png")
        with open(mask_filename,'wb') as f:
            mask_image.save(f)
        return mask_image

    def get_staff_info(self, mask_image, temp_dir):
        logging.info("get_staff_info %s", temp_dir)
        staff_info = []
        mask_labels, num_labels = get_mask_labels(mask_image)
        threshold_size = mask_image.height * mask_image.width * 0.005
        mask_rects = get_good_mask_rects(mask_labels, num_labels, threshold_size)
        save_staff_images(temp_dir, self.score_image, mask_rects, staff_info)
        staff_info_filename = os.path.join(temp_dir, "staff_info.json")
        with open(staff_info_filename, 'w') as outfile:
            json.dump(staff_info, outfile)
        return staff_info

    def scan_staff_info(self, staff_info, key):
        """Return music_info from decoding the staff images"""
        music_info = []
        for cur_staff_info in staff_info:
            cur_crop_staff_inputs = get_crop_inputs(cur_staff_info)
            key_inputs = np.zeros((cur_crop_staff_inputs.shape[0],NUM_KEY_BITS))
            key_vec = np.array(list(np.binary_repr(key,width=NUM_KEY_BITS)),dtype=np.float32)
            key_inputs = key_inputs + key_vec
            note_preds, length_preds = self.scan_model.predict([cur_crop_staff_inputs,
                                                                key_inputs])
            raw = get_notes_lengths(note_preds, length_preds)
            refined = refine_notes_lengths(raw)
            music_info.extend(refined)
        return music_info

    def output_music(self, music_info, output_filename):
        output_rmf(output_filename, music_info)

    def parse_args(self,argv):
        """parse commandline arguments, use config files to override default
        values. Initializes self.args: a dictionary of your
        commandline options
        """
        parser = argparse.ArgumentParser(prog='romer',
                                         description="Read sheet music.")
        parser.add_argument(
            "-v","--verbose",
            dest="verbose",
            action='count',
            default=self.config.get("options","verbose",0),
            help="Increase verbosity (add once for INFO, twice for DEBUG)"
        )
        parser.add_argument(
            "--model_dir",
            dest="model_dir",
            default=self.config.get("options","model_dir","./models"),
            help="Specify where the models & weights are found."
        )
        parser.add_argument(
            "--temp_dir",
            dest="temp_dir",
            default=self.config.get("options","temp_dir","./temp"),
            help="Specify where temporary, intermediate files may be stored."
        )
        parser.add_argument(
            "-i", required=True,
            dest="input_filename",
            help="Required input image."
        )
        parser.add_argument(
            "-o", required=True,
            dest="output_filename",
            help="Required output file. (rmf format)"
        )
        parser.add_argument(
            "-k", "--key",
            dest="key",
            default="7",
            help="[temp until we can figure this out] key value: 0-14 (default=7=C)"
        )
        self.args = parser.parse_args(argv)


    def adjust_logging_level(self):
        """adjust logging level based on verbosity option
        """
        log_level = logging.WARNING # default
        if self.args.verbose == 1:
            log_level = logging.INFO
        elif self.args.verbose >= 2:
            log_level = logging.DEBUG
        logging.basicConfig(level=log_level)

# ======================================================================
class Config(object):
    """Config Class.  Use a configuration file to control your program
    when you have too much state to pass on the command line.  Reads
    the romer.cfg or ~/.romer.cfg file for configuration options.
    Handles booleans, integers and strings inside your cfg file.
    """
    def __init__(self,program_name=None):
        self.config_parser = configparser.ConfigParser()
        if not program_name:
            program_name = os.path.basename(sys.argv[0].replace('.py',''))
        self.config_parser.read([program_name+'.cfg',
                                 os.path.expanduser('~/.'+program_name+'.cfg')])
    def get(self,section,name,default):
        """returns the value from the config file, tries to find the
        'name' in the proper 'section', and coerces it into the default
        type, but if not found, return the passed 'default' value.
        """
        try:
            if type(default) == type(bool()):
                return self.config_parser.getboolean(section,name)
            elif type(default) == type(int()):
                return self.config_parser.getint(section,name)
            else:
                return self.config_parser.get(section,name)
        except:
            return default

# ======================================================================
def main(argv):
    """ The main routine creates and runs the Application.
    argv: list of commandline arguments without the program name
    returns application run status
    """
    app = Application(argv)
    return app.run()

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
