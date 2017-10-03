# -*- mode: makefile; -*-
all:
	make -f Makefile0 # generate gen_*.ly files
	make -f Makefile1 # run lilypond & inkscape to create svg, midi and png files

clean:
	make -f Makefile1 clean
	make -f Makefile0 clean
