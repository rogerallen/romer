#!/usr/bin/env python3
"""
skeleton.py - a skeleton starting-point for python scripts by Roger Allen.

Any copyright is dedicated to the Public Domain.
http://creativecommons.org/publicdomain/zero/1.0/
You should add your own license here.

"""
import os
import sys
import logging
import argparse
import configparser

sys.path.insert(0, "..")
import romer

def diff(filename0, filename1):
    with open(filename0,'r') as f0:
        with open(filename1,'r') as f1:
            line_num = 1
            for line0 in f0:
                if line0.startswith("key"):
                    continue
                try:
                    line1 = next(f1)
                    if line0 != line1:
                        print(f"FAIL: files {filename0} and {filename1} differ on line {line_num}")
                        return True
                except StopIteration:
                    print(f"FAIL: files {filename0} and {filename1} differ on line {line_num}")
                    return True
                line_num += 1

    print(f"PASS: files {filename0} and {filename1} are the same")
    return False

# ======================================================================
def test_diff(infile,outfile,goldfile,key=7):
    print(f"----------------------------------------------------------------------\ntesting {infile}")
    romer.main(["--model_dir", "../models", "-i", infile, "-o", outfile, "-k", str(key)])
    diff(goldfile, outfile)

def test_setup_dir():
    setup_files = [("../setup/chromatic.png", "chromatic.rmf","../setup/chromatic.rmf"),
                   ("../setup/twinkle.png", "twinkle.rmf", "../setup/twinkle.rmf"),
                   ("../setup/gen_one.png", "gen_one.rmf", "../setup/gen_one.rmf"),
                   ("../setup/gen_two.png", "gen_two.rmf", "../setup/gen_two.rmf"),
                   ("../setup/gen_three_1.png", "gen_three_1.rmf", "../setup/gen_three_1.rmf"),
                   ("../setup/gen_three_2.png", "gen_three_2.rmf", "../setup/gen_three_2.rmf"),
                   ("../setup/gen_three_3.png", "gen_three_3.rmf", "../setup/gen_three_3.rmf")]
    for infile,outfile,goldfile in setup_files:
        test_diff(infile,outfile,goldfile)

def test_setup_keys():
    test_diff("../setup/frere.png", "frere.rmf", "../setup/frere.rmf", 6)
    for i,k in enumerate("g d a e b fis cis".split()):
        test_diff(f"../setup/gen_key_{k}.png", f"gen_key_{k}.rmf", f"../setup/gen_key_{k}.rmf", i+1+7)
    for i,k in enumerate("f bes ees aes des ges ces".split()):
        test_diff(f"../setup/gen_key_{k}.png", f"gen_key_{k}.rmf", f"../setup/gen_key_{k}.rmf", 7-1-i)

def test_frere():
    freres = [("Frère/320px-Frère_Jacques.svg.png", "frere_320px.rmf"),
              ("Frère/640px-Frère_Jacques.svg.png", "frere_640px.rmf"),
              ("Frère/800px-Frère_Jacques.svg.png", "frere_800px.rmf"),
              #("Frère/1024px-Frère_Jacques.svg.png", "frere_1024px.rmf")
    ]
    for infile,outfile in freres:
        test_diff(infile,outfile,"Frère/Frère.rmf")

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
        test_setup_keys()
        test_setup_dir()
        test_frere()
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
        # more args https://docs.python.org/3/library/argparse.html
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
