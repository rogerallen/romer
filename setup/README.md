# Romer Training Data

## Goals

The goal of Romer is to take an input sheet music image and output a midi file or other file that can be used to synthesize sound that corresponds to that sheet music.

The goal of the code in this area is to create files that make that possible.  We will use two Free and Open Source applications, Lilypond and Inkscape along with some scripts in the src directory.

## Usage

You will need Python 3.6, Lilypond 2.18.2 and Inkscape 0.92.2 to reproduce my environment.  Other versions may work.  My intent is that this should work on Windows, Linux and Mac OS X.

Add score files to this directory as filename.ly or add generated scores via the src/gen_test_scores.py script.  These files should have a name starting with "gen_".  I suspect we are going to need lots and lots of example music!

Create the rest of the output via the make_files.py script.  It will run all the programs to create the files.

## To Do

- [x] ~~try windows (this was coded on the mac first)~~
- [x] ~~try linux~~
- [x] ~~align x,y between rmfxy & png~~
- [x] first attempt: train & learn with this data
- [ ] generate music in all clefs
- [ ] generate music in all key signatures
- [ ] generate music in all time signatures (actually, do I care?)
- [ ] generate music with multiple notes per beat.
- [ ] generate multi-part scores (piano staff to start)
- [ ] generate pngs at different sizes (vary inkscape DPI)
- [ ] what about repeats?
- [ ] get free scores
  - [ ] https://veltzer.github.io/openbook/, https://github.com/veltzer/openbook
  - [ ] http://www.mutopiaproject.org/, https://github.com/MutopiaProject/MutopiaProject
  - [ ] find others
- [ ] fully train & learn with this data

## Details

First, we will start with a score file that describes the music. Lilypond will be used to translate that score file into a few different files: an svg file that can be used to create the score image, a midi file that could be used to synthesize the score, and finally a notes file that describes the rendered score in an easy to parse fashion.

Next, we'll use Inkscape to create a png image of the sheet music using the svg file as input.

At this point, we theoretically have all we need to train a neural network: a png image file and a midi file.  Go for it! Good luck!

I suspect we need to help a little bit more before it will work.  The nice thing is that Lilypond has helped us greatly with hints in both the svg file and the notes file that point back to the original score file, allowing us to directly tie together the image to the musical notes.

To help, what I've done is create a few more files to use for training.

First, by parsing the svg file, we can easily find the 5 lines that draw the staff.  Given that, we can output rectangles that show exactly where a neural network should pay attention.  The gen_mask.py script outputs an svg with rectangles that bound the staff lines and inkscape is used to render that to a png image file.

Next, by taking both the lilypond svg file and the notes file, we can create a 'rmfxy' text file that describes each note on a line and along with that note an x,y location for that note in the image.  It also outputs an 'rmf' file without that x,y location for direct comparison to what the model outputs.  It is an easy format for describing music.

At this point, you have a png image file, an output file that describes the notes to synthesize and a way to tie those notes to locations within the image file.  My hope is that this will allow for real progress.

### Notes file

The standard event-listener.ly file from Inkscape provided a pathway to map notes to the SVG image.  To properly account for note start times, we'll also want to account for rests.  So, we'll use our own event-listener.ly file--event-romer.ly.

The original file is from /usr/share/lilypond/2.18.2/ly on my Linux system.

Rests will be called "notes" in the .rmf files, but the pitch of the note will be -1

## Example

Here is an example of "Twinkle, Twinkle, Little Star" and what each file contains.

- twinkle.ly: this is the starting point score file describing the song for Lilypond.

When Lilypond runs, it outputs three files:
- twinkle.svg: a description of the sheet music.
- twinkle.midi: a file that can be used directly to synthesize music.
- twinkle-unnamed-staff.notes: a text file with details about each note in the score.  This is output due to the `\include "event-listener.ly"` in the score file.

Now gen_mask.py can run and find the staff lines to create a mask file:
- twinkle_mask.svg

Inkscape can use both svg files to create images.
- *twinkle.png*: the actual score sheet music image
- twinkle_mask.png: a mask to highlight where the staff lines are.

Finally, gen_rmfxy outputs both of these files:
- *twinkle.rmfxy*: notes along with their x,y locations in the score svg
- twinkle.rmf: just the notes

With these files generated we have the two main files to use for training.  The image, of course is *twinkle.png*.  The music and notes description file *twinkle.rmfxy* has the following format, one note per line.

```
note,<midi pitch>,<start beat>,<duration in beats>,<note image x>,<note image y>
```

You can see this illustrated with a few notes from the beginning of "Twinkle, Twinkle Little Star"

![Image of Twinkle notes and score](https://github.com/rogerallen/romer/raw/master/doc/twinkle_note_png.jpg)
