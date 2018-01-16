# Romer

The goal of Romer is to take an input sheet music image and output a midi file or other file that can be used to synthesize sound that corresponds to that sheet music.

## Getting Started

This is barely able to run testcases.  Lower your expectations.

0. you will want a python virtual environment setup.  This works for running romer.py.
```
conda create --name romer
activate romer
conda install scipy numpy matplotlib scikit-learn scikit-image pillow h5py
pip install tensorflow-gpu
pip install keras
```
1. Create test data in the setup directory.  See the README file.
2. Train your model in the train directory.  See the README file.
3. Copy or link your models & weights into the models directory:
   length_model.json  mask_model.json  note_model.json
   length_weights.h5  mask_weights.h5  note_weights.h5
4. Run romer.py on an image.  The output rmf file should match the one from the setup directory.  Something like:

   ./romer.py -i setup/twinkle.png -o twinkle.rmf

## Usage

We'll get there eventually.

## License

This project is licensed under the GPL V3 License - see https://www.gnu.org/licenses/gpl-3.0.en.html for details

## Acknowledgments

* http://www.fast.ai/
* https://www.meetup.com/Portland-Deep-Learning/
