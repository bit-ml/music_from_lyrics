# Bitdefender 2017

import os
import torch
import codecs
import sys
import gensim
import pdb
import torchwordemb
import pickle


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("give filenames please!")
    print("loading...")
    vocab, vec = torchwordemb.load_glove_text(sys.argv[1])
    print("saving...")
    with open(sys.argv[2], 'wb') as output:
        pickle.dump(vocab, output, -1) 
    torch.save(vec, sys.argv[3])
    print("done!")


