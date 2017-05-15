# Bitdefender 2017

import os
import torch
import codecs
import sys
import gensim
import pdb
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

mn = sys.argv[1]

errs = torch.load(mn).numpy()

last_ix = -1
for i in range(len(errs)):
    if errs[i] < 0.0005:
        last_ix = i
        break

errs = errs[:last_ix]

fg = plt.figure()
plt.plot(errs)
fg.suptitle(mn +  ' error', fontsize=20)
plt.xlabel('batch groups', fontsize=16)
plt.ylabel('error', fontsize=16)
fg.savefig(mn + '.jpg')

