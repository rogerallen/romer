#!/usr/bin/env python3
"""
make_files.py - because I can't assume make on windows is a thing.
"""

import argparse
import configparser
import glob
import logging
import os
import subprocess
import sys

import src.gen_test_scores as gts
import src.gen_mask as gm
import src.gen_rmfxy as grmf

if os.name == 'nt':
    MIDI_SUFFIX = '.mid'
else:
    MIDI_SUFFIX = '.midi'

def getmtime(path):
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return 0.0

# ======================================================================
def generate_test_lilypond_files():
    exe_mtime = getmtime('src/gen_test_scores.py')
    out_mtime = getmtime('gen_one.ly')
    if exe_mtime > out_mtime:
        gts.main()
    else:
        logging.info("generate_test_lilypond_files: nothing to generate.")

def run_lilypond_one(args,src):
    lilypond_exe = 'lilypond'
    if args.lilypond_path != '':
        lilypond_exe = os.path.join(args.lilypond_path, lilypond_exe)
    notes = src.replace('.ly','-unnamed-staff.notes')
    logging.info("run_lilypond_one: removing %s"%(notes))
    try:
        os.remove(notes)
    except FileNotFoundError:
        pass
    cmd = [lilypond_exe,'-dbackend=svg',src]
    logging.info("run_lilypond_one: running %s"%(' '.join(cmd)))
    cp = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
    print(cp.stdout)

def run_lilypond(args):
    sources = glob.glob('*.ly')
    for src in sources:
        src_mtime = getmtime(src)
        dests = [src.replace('.ly','.svg'),
                 src.replace('.ly',MIDI_SUFFIX),
                 src.replace('.ly','-unnamed-staff.notes')]
        dest_mtime = min([getmtime(p) for p in dests])
        if src_mtime > dest_mtime:
            run_lilypond_one(args,src)
        else:
            logging.info('run_lilypond: nothing to generate for %s'%(src))

def create_mask_svgs():
    sources = glob.glob('*.ly')
    exe_mtime = getmtime('src/gen_mask.py')
    for src in sources:
        src = src.replace('.ly','.svg')
        dest = src.replace('.svg','_mask.svg')
        src_mtime = min(exe_mtime,getmtime(src))
        dest_mtime = getmtime(dest)
        if src_mtime > dest_mtime:
            gm.main(src,dest)
        else:
            logging.info('create_mask_svgs: nothing to generate for %s'%(src))

def run_inkscape_one(args,src,dest):
    src_path = os.path.join(os.getcwd(),src)
    dest_path = os.path.join(os.getcwd(),dest)
    cmd = [os.path.join(args.inkscape_path,"inkscape"),
           "--export-png=%s"%(dest_path),
           "--export-dpi=96",
           src_path]
    logging.info("run_inkscape_one: running %s"%(' '.join(cmd)))
    cp = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
    print(cp.stdout)

def run_inkscape(args):
    sources = glob.glob('*.svg')
    for src in sources:
        dest = src.replace('.svg','.png')
        src_mtime = getmtime(src)
        dest_mtime = getmtime(dest)
        if src_mtime > dest_mtime:
            run_inkscape_one(args,src,dest)
        else:
            logging.info('run_inkscape: nothing to generate for %s'%(src))

def generate_rmfxy_files():
    sources = glob.glob('*-unnamed-staff.notes')
    for src in sources:
        src1 = src.replace('-unnamed-staff.notes','.svg')
        src_mtime = max([getmtime(src),getmtime(src1)])
        dest = src.replace('-unnamed-staff.notes','.rmf')
        dest1 = src.replace('-unnamed-staff.notes','.rmfxy')
        dest_mtime = min([getmtime(dest),getmtime(dest1)])
        if src_mtime > dest_mtime:
            grmf.main(src,src1,dest.replace('.rmf',''))
        else:
            logging.info('run_lilypond: nothing to generate for %s'%(src))


def clean_files():
    sources = glob.glob('*.ly')
    dests = glob.glob('gen*.ly')
    for src in sources:
        dests += [
            src.replace('.ly',MIDI_SUFFIX),
            src.replace('.ly','.png'),
            src.replace('.ly','.svg'),
            src.replace('.ly','_mask.png'),
            src.replace('.ly','_mask.svg'),
            src.replace('.ly','-unnamed-staff.notes'),
            src.replace('.ly','.rmf'),
            src.replace('.ly','.rmfxy')
        ]
    logging.info("clean_files: %s"%(dests))
    for d in dests:
        try:
            os.remove(d)
        except FileNotFoundError:
            pass

# ======================================================================
class Application(object):
    def __init__(self,argv):
        self.config = Config()
        self.parse_args(argv)
        self.adjust_logging_level()

    def run(self):
        """The Application main run routine
        """
        # -v to see info messages
        logging.info("Args: {}".format(self.args))
        if self.args.clean:
            clean_files()
            return 0
        generate_test_lilypond_files()
        run_lilypond(self.args)
        create_mask_svgs()
        run_inkscape(self.args)
        generate_rmfxy_files()
        return 0

    def parse_args(self,argv):
        """parse commandline arguments, use config files to override
        default values. Initializes:
        self.args: a dictionary of your commandline options,
        """
        parser = argparse.ArgumentParser(description="A python3 skeleton.")
        parser.add_argument(
            "-v","--verbose",
            dest="verbose",
            action='count',
            default=self.config.get("options","verbose",0),
            help="Increase verbosity (add once for INFO, twice for DEBUG)"
        )
        parser.add_argument(
            "--clean",
            action='store_true',
            help="Clean all files created by this script."
        )
        parser.add_argument(
            "--lilypond_path",
            dest="lilypond_path",
            default=self.config.get("lilypond","path",""),
            help="specify path to lilypond executable"
        )
        parser.add_argument(
            "--inkscape_path",
            dest="inkscape_path",
            default=self.config.get("inkscape","path",""),
            help="specify path to inkscape executable"
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
    when you have too much state to pass on the command line.
    Reads the <program_name>.cfg or ~/.<program_name>.cfg file for
    configuration options.
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
