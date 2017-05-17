# Bitdefender 2017

import os.path
import os
import sys
from subprocess import call


if __name__=="__main__":
    if len(sys.argv) < 3:
        print("please give sentiment and folder in command line!")
    sentiment = sys.argv[1]
    l = [x for x in os.listdir(sys.argv[2]) if sentiment in x and "mel" in x]
    for ix,v in enumerate(l):
        string = os.path.join(sys.argv[2], sentiment + "_" + str(ix) + ".mid")
        v = os.path.join(sys.argv[2], v)
        call(["abc2midi", v, "-v", "-o", string])

