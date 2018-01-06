#!/bin/env python
from random import choice, seed, shuffle

ly_template_start = """\\include "event-listener.ly"
\\paper {
  #(set-paper-size "letter")
}
\\score {
  {
    \\clef treble
    \\numericTimeSignature
    \\time 4/4"""

ly_template_end = """  }
  \\layout { }
  \\midi { }
  \\version "2.18.2"
}"""

def gen_test_one():
    name = "gen_one.ly"
    print("generating: %s"%(name))
    pitches = "c cis des d dis ees e f fis ges g gis aes a ais bes b r".split()
    octaves = [ "'", "''" ]
    lengths = [ "1", "2", "4", "8", "16" ]
    with open(name,'w') as f:
        print(ly_template_start,file=f)
        count = 0
        for l in lengths:
            for o in octaves:
                for p in pitches:
                    if p != 'r':
                        print("%s%s%s"%(p,o,l),end=' ',file=f)
                    else:
                        print("%s%s"%(p,l),end=' ',file=f)
                    count += 1
                    if count%8==0:
                        print(file=f)
        print(ly_template_end,file=f)

def gen_test_two():
    seed(1234)  # random, but deterministic behavior
    name = "gen_two.ly"
    print("generating: %s"%(name))
    pitches = "c cis des d dis ees e f fis ges g gis aes a ais bes b r".split()
    octaves = [ "'", "''" ]
    lengths = ["4"]*4 + ["8"]*4 + ["2", "1"]
    [ "1", "2", "4", "8" ]
    with open(name,'w') as f:
        print(ly_template_start,file=f)
        count = 0
        li = 0
        for i in range(8):
            shuffle(pitches)
            for o in octaves:
                for p in pitches:
                    if p != 'r':
                        print("%s%s%s"%(p,o,lengths[li]),end=' ',file=f)
                    else:
                        print("%s%s"%(p,lengths[li]),end=' ',file=f)
                    count += 1
                    if count%8==0:
                        print(file=f)
                    li = (li+1)%len(lengths)
        print(ly_template_end,file=f)

def main():
    gen_test_one()
    gen_test_two()

if __name__ == "__main__":
    main()
