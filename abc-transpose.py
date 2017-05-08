# /bin/python

# Bitdefender 2017
# Modified from:
# http://stackoverflow.com/questions/23904552/crawling-this-website-to-grab-lyrics


'''There are times when you want to transpose an abc file and are
far too lazy to transpose them yourself. This is a python script you
can use to transpose your files.

USAGE:
abc-transpose.py [transpose] /path/to/input/file /path/to/output/file

Where [transpose] is one of the following:
-u [number of semi-tones]
-d [number of semi-tones]
'''

import sys, getopt
import pdb

class Transposer:

    def __init__(self, args):
        self.TRANSPOSE = ['C,', 'D,', 'E,', 'F,', 'G,', 'A,', 'B,', "C", "D",
                     "E", "F", "G", "A", "B", "c", "d", "e", "f", "g",
                     "a", "b", "c'", "d'", "e'", "f'", "g'", "a'", "b'"]
        TRANSPOSE_UP = []
        for note in self.TRANSPOSE[self.TRANSPOSE.index("c'"):]:
            TRANSPOSE_UP.append(note + "'")
        TRANSPOSE_DOWN = []
        for note in self.TRANSPOSE[:self.TRANSPOSE.index("C,")]:
            TRANSPOSE_DOWN.append(note + ",")
        last_arg = self.process_args(args)
        if not(self.infile and self.outfile):
            if self.outfile == -1:
                self.infile = args[-1]
            elif last_arg == args[-1]:
                print("Need to specify the input and output files")
                sys.exit(2)
            elif last_arg == args[-2]:
                self.infile = args[-1]
                self.outfile = -1
            else:
                self.infile = sys.argv[-2]
                self.outfile = sys.argv[-1]
        if self.transpose > 0:
            self.TRANSPOSE.append(TRANSPOSE_UP)
        else:
            self.TRANSPOSE.append(TRANSPOSE_DOWN)

    def process_args(self, args):
        self.transpose = 0
        self.infile = 0
        self.outfile = 0
        self.number = 0
        self.debug = False
        UP_OPTS = ["-u", "--up"]
        DOWN_OPTS = ["-d", "--down"]
        IN_OPTS = ["-i", "--input"]
        OUT_OPTS = ["-o", "--output"]
        CONSOLE_OPTS = ["-c", "--console"]
        NUMBER_OPTS = ["-n", "--number"]
        DEBUG_OPTS = ["--debug"]
        try:
            # args = " ".join(args)
            opts, args = getopt.getopt(args,
                                       "cu:d:i:o:n:",
                                       ["up=", "down=", "input=",
                                        "output=", "console",
                                        "number=", "debug"])
            # print(args)

            # opts, args = getopt.getopt(args, "d")
            print(opts)

            # print("bubu are mere")
            # print(opts)
            # print(args)
            # print(opts)
            for opt, arg in opts:
                # print("gigel")
                if opt in UP_OPTS:
                    if self.transpose:
                        print("Already passed a value")
                        raise ValueError("Already passed a value")
                    self.transpose = int(arg)
                elif opt in DOWN_OPTS:
                    if self.transpose:
                        print("Already passed a value")
                        raise ValueError("Already passed a value")
                    self.transpose = -int(arg)
                elif opt in IN_OPTS:
                    self.infile = arg
                elif opt in OUT_OPTS:
                    self.outfile = arg
                elif opt in CONSOLE_OPTS:
                    self.outfile = -1
                elif opt in NUMBER_OPTS:
                    self.number = int(arg)
                elif opt in DEBUG_OPTS:
                    self.debug = True
                last_arg = arg
            return last_arg
        except:
            # sys.exc_info()[0]
            self.usage()
            sys.exit(2)

    def usage(self):
        print()
        print("USAGE:")
        print("abc-transpose.py [transpose] /path/to/input /path/to/output")
        print("")
        print("Where [transpose] is one of the following:")
        print("    -u [number of semi-tones]")
        print("    -d [number of semi-tones]")
        print("OPTIONS:")
        print("-u | --up      : Number of semi-tones to increase by")
        print("-d | --down    : Number of semi-tones to decrease by")
        print("-i | --input   : The input file")
        print("-o | --output  : The output file")
        print("-c | --console : Outputs to terminal. Default output file")
        print("-n | --number  : The number in the file to transpose")
        print("--debug        : Get debugging information")
        print()

    def description(self, line):
        MARKERS = "ABCDEFGHIJLMNOPQRSTUVWXYZabcdefghijlmnopqrstuvwxyz"
        for marker in MARKERS:
            if (marker + ":") in line:
                return True
        stripped_line = line.strip()
        if len(stripped_line) > 0 and stripped_line[0] == "%":
            return True
        return False

    def transpose_key(self, line):
        keys = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A",
                "Bb", "B"]
        scales = ["maj", "min", "dor", "phr", "lyd", "mix", "aol"]
        key = False
        for scale in scales:
            if scale in line:
                key = line[line.rfind(":")+1:line.rfind(scale)]
                used_scale = scale
        if not key:
            line = line + "maj"
            key = line[line.rfind(":")+1:line.rfind("maj")]
            used_scale = "maj"
        new_key = keys[(keys.index(key.strip()) + self.transpose) % len(keys)]
        return "K: " + new_key + line[line.rfind(used_scale):]

    def transpose_note(self, note):
        if self.debug:
            print("Transposing " + note)
        shift = self.transpose // 2
        if note in self.TRANSPOSE:
            note = self.TRANSPOSE[(self.TRANSPOSE.index(note) + shift)]
            if self.debug:
                print("Note found, transposed to " + note)
        return note

    def transpose_line(self, line):
        buff = ""
        to_return = ""
        comment = False
        for character in line:
            if character == "%":
                comment = True
                to_return += self.transpose_note(buff)
                buff = character
            elif comment:
                buff += character
            elif character in ["'", ","]:
                buff = character
            else:
                to_return += self.transpose_note(buff+character)
                buff = ""
        return to_return + buff

    def song_check(self, line):
        return self.number == 0 or int(line[line.rfind(":")+1:].strip()) == self.number

    def transpose_file(self):
        if self.debug:
            print("Beginning tranpose of " + self.infile)
        to_return = ""
        with open(self.infile) as infile:
            do_song = True
            for line in infile:
                if "X:" in line or "x:" in line:
                    do_song = self.song_check(line)
                if do_song and not self.description(line):
                    if "K:" in line or "k:" in line:
                        out = self.transpose_key(line)
                    else:
                        out = self.transpose_line(line)
                    to_return += out
                else:
                    to_return += line
        return to_return
                    
    def write_output(self, output):
        if self.outfile == -1:
            print(output.strip())
        else:
            with open(self.outfile, "a") as outfile:
                outfile.write(output)
        
if __name__ == "__main__":
    args = sys.argv[1:]
    transposer = Transposer(args)
    transposed = transposer.transpose_file()
    transposer.write_output(transposed)
