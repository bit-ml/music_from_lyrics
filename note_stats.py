import os
import torch
import codecs
import sys
import gensim
import pdb
import torchwordemb
import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt


class Stats_Retriever(object):
    def __init__(self):
        self.exclusion_dict = set(["X:", "T:", "%", "K:", "P:", "M:",\
                "S:", "N:", "C:", "Z:", "D:", "I:","R:","L","Q:", "B:",\
                "O:","G:","H:","W:", "A:"])



    def exclude(self, line):
        if line.isspace():
            return True
        for ex in self.exclusion_dict:
            if line.startswith(ex):
                return True
        return False

    def get_raw_frequency(self, file_name, freq):
        with open(file_name, 'r') as f:
            for line in f:
                if self.exclude(line):
                    continue
                else:
                    for c in line.strip():
                        if c in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                            if c not in freq:
                                freq[c] = 1
                            else:
                                freq[c] += 1
        return freq



    def get_frequencies(self, file_name):
        freq = {}
        if os.path.isdir(file_name):
            for f in os.listdir(file_name):
                freq = self.get_raw_frequency(f, freq)
        else:
            freq = self.get_raw_frequency(file_name, freq)

        all_notes = 0
        for f in freq:
            all_notes += freq[f]

        for f in freq:
            freq[f] = float(freq[f]) / float(all_notes)

        d = sorted(freq)
        t = torch.Tensor(len(d))

        for i,k in enumerate(d):
            t[i] = freq[k]

        return freq


    def plot_histogram(self, f, show_histogram=False, save_plot_path=None):
        plt.bar([x for x in range(7)], f.values())
        plt.title('Note frequencies')
        plt.xlabel('Note names')
        plt.xticks(list(range(7)), ('A', 'B', 'C', 'D', 'E', 'F', 'G'))
        if save_plot_path:
           plt.savefig(save_plot_path)
           print("saved to " + save_plot_path)
        if show_histogram:
           plt.show()

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("give file/folder name")
        sys.exit(-1)

    sr = Stats_Retriever()
    freqs = sr.get_frequencies(sys.argv[1])
    sr.plot_histogram(freqs, show_histogram=True, save_plot_path="stats_trist.png")


