# Romer Training

## Goals

The goal of Romer is to take an input sheet music image and output a midi file or other file that can be used to synthesize sound that corresponds to that sheet music.

The goal of the code in this area is to use the files created in the ../data directory to train a model we design and maintain in this directory.  We will use TBD deep learing libraries.

## Usage

1. step through the 'mask_data_prep.ipynb' to create the training,
validation & test set of images to train the masking model.

2. run train_mask_model.py to create the model & model weights.

3. test the model & create masks with the 'check_mask_model.ipynb' notebook.

4. run 'score_data_prep.ipynb' to setup data/train2 and data/train3 with the selected score staff lines.

5. run train_score_model.py to create the note & lengths models & model weights.

6. run check_score_model.ipynb and compare data/results/*rmf with setup/*rmf

## To Do

- [X] come up with a model for masking regions of interest
- [X] train model for masking regions of interest
- [X] come up with a CNN model for scanning staff lines
- [X] handle different key signatures
- [ ] code cleanup: rename note->pitch and score->staff as appropriate
- [ ] handle different clefs
- [ ] handle chords

## Architecture

Just ideas at this point...

Input is:
- a full image of the score

Output is a sequence of notes.  For now a text file output.
- time to play this note
  - perhaps time-since-last-note is easier for network
- note pitch
- note duration
- caveat: a common occurrence is to play multiple notes at the same time.

Data Pipeline:
- align & clean up input images
- direct attention to specific regions of the image (staff lines)
- need to scan across the staff lines, showing a portion of the overall image
  - stateful information:
    - clef
    - time signature
    - time (so you can tell what time your note should play)
    - previously output note (for notes in chords, at least)
      - if we had N note-detectors where N ~ 5-10, that might work, too. "polyphony"
      - our perhaps "multipass" with count of notes to output & current output count
    - was previous staff connected? (e.g. piano grand staff)
  - not stateful
    - note name & duration when centered in the image
    - number of notes
    - information about where notes are in the image to help scanning operation
      notes-to-the-right, notes-to-the-left, notes-at-center

Output Encoding:
