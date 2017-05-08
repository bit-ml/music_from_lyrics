import os
import torch
import codecs
import sys
import gensim
import pdb
import subprocess
import os
from os import listdir
from os.path import isfile, join


exec_name = "abc2abc"

maj1 = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
maj2 = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


majors = [maj1, maj2,
        [x + " Major" for x in maj1],\
        [x + " Major" for x in maj2],\
        [x + " Maj" for x in maj1],\
        [x + " Maj" for x in maj1]]


min1 = ['Am','Bbm','Bm','Cm','Dbm','Dm','Ebm','Em','Fm','Gbm','Gm','Abm']
min2 = ['Am','A#m','Bm','Cm','C#m','Dm','D#m','Em','Fm','F#m','Gm','G#m']

minors = [min1, min2, [x + " Minor" for x in min1],\
        [x + " Minor" for x in min2],
        [x + " Min" for x in min1],\
        [x + " Min" for x in min2]]



def key(key_line):
    key = key_line[key_line.strip().find(":") + 1:]
    for mlist in majors:
        for k,m in enumerate(mlist):
            if m == key or m.lower() == key.lower():
                return k,m

    for mlist in minors:
        for k,m in enumerate(mlist):
            if m == key or m.lower() == key.lower():
                return k,m

    return None, None

def transpose(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    index = 0
    if not os.path.exists("transposed/"):
        os.makedirs("transposed/")
    for f in files:
        f_path = join(path,f)
        with open(f_path, 'r', encoding="ISO-8859-1") as of:
            content = of.read().strip()
            songs = content.split("X:")
            for song in songs:
                ix = song.find("K:")
                if ix == -1:
                    continue
                ix2 = song[ix + 1:].find("K:")
                if ix2 != -1:
                    song = song[:ix2]
                key_line = song[ix:].split('\n')[0].strip()
                k,m = key(key_line)
                if k is None:
                    continue
                index += 1
                temp = open("transposed/temp.abc", "w")
                song = "X: " + str(index) + "\n" + song
                temp.write(song)
                temp.close()
                args =  [exec_name]
                args += ["transposed/temp.abc"]
                args += ["-t"]
                args += ["-" + str(k)]
                my_env = os.environ.copy()
                popen = subprocess.Popen(args, stdout=subprocess.PIPE,\
                        env=my_env)
                popen.wait()
                output = popen.stdout.read().decode("latin-1")
                new_file = open("transposed/" + str(index) + ".abc", "w")
                new_file.write(output)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        assert False, "need name of folder with files"

    path = sys.argv[1]
    transpose(path)


