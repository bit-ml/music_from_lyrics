import os
import torch
import codecs
import sys
import gensim
import pdb
import torchwordemb
import pickle


class Dictionary_music(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = []
    def add_char(self, char):
        if char not in self.char2idx:
            self.idx2char.append(char)
            self.char2idx[char] = len(self.idx2char) - 1
        return self.char2idx[char]

    def __len__(self):
        return len(self.idx2char)


class Dictionary_words(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus_music(object):
    def __init__(self, path):
        self.dictionary = Dictionary_music()
        self.exclusion_dict = set(["X:", "T:", "%", "K:", "P:", "M:",\
                "S:", "N:", "C:", "Z:", "D:", "I:","R:","L","Q:", "B:",\
                "O:","G:","H:","W:", "A:"])

        self.train_j = self.tokenize(os.path.join(path, 'major.abc'))
        self.train_w = self.tokenize(os.path.join(path, 'minor.abc'))


    def exclude(self, line):
        if line.isspace():
            return True
        for ex in self.exclusion_dict:
            if line.startswith(ex):
                return True
        return False

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                if self.exclude(line):
                    continue
                chars = list(line)
                tokens += len(chars)
                for char in chars:
                    self.dictionary.add_char(char)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                if self.exclude(line):
                    continue
                chars = list(line)
                for char in chars:
                    ids[token] = self.dictionary.char2idx[char]
                    token += 1

        return ids


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
        import os.path
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
        import numpy as np
        import matplotlib.pyplot as plt
        plt.bar([x for x in range(7)], f.values())
        plt.title('Note frequencies')
        plt.xlabel('Note names')
        plt.xticks(list(range(7)), ('A', 'B', 'C', 'D', 'E', 'F', 'G'))
        if save_plot_path:
           plt.savefig(save_plot_path)
           print("saved to " + save_plot_path)
        if show_histogram:
           plt.show()


class Corpus_words(object):
    def __init__(self, path, pretrained_vocab=None, pretrained_embs=None):
        corpus_txt = None
        corpus_a = None
        corpus_b = None

        corpus_path = os.path.join(path, "all.txt")
        a_path = os.path.join(path, "thrash.txt")
        b_path = os.path.join(path, "pop.txt")

        with open(corpus_path, "r") as f:
            corpus_txt = f.read()

        with open(a_path, "r") as f:
            corpus_a = f.read()

        with open(b_path, "r") as f:
            corpus_b = f.read()

        l_txt = [x.split(" ") for x in\
                corpus_txt.strip().split("\n") if len(x) > 0]

        vocab = None
        vec   = None

        if not pretrained_vocab:
            model = gensim.models.Word2Vec(l_txt, size=200, min_count=1)
            vocab = model.wv.vocab
            vec = torch.from_numpy(model.wv.syn0)
        else:
            try:
                with open(pretrained_vocab, "rb") as inp:
                    vocab, vec = pickle.load(inp), torch.load(pretrained_embs)
            except IOError:
                print("preloaded vocab not found. " + \
                        " run quick_save.py with the embeddings text file")
                sys.exit(-1)

        self.embeddings = vec

        self.all_dictionary = Dictionary_words()

        self.a_dictionary = Dictionary_words()
        self.b_dictionary = Dictionary_words()


        for k in vocab:
            self.all_dictionary.add_word(k)


        a_corpus = [x.split(" ") for x in\
                corpus_a.strip().split("\n") if len(x) > 0]


        b_corpus = [x.split(" ") for x in\
                corpus_b.strip().split("\n") if len(x) > 0]

        tokens_a = 0
        for s in a_corpus:
            for w in s:
                if w in self.all_dictionary.word2idx:
                    tokens_a += 1
                    self.a_dictionary.add_word(w)

        self.a_idxs = torch.zeros(tokens_a).long()
        ix = 0
        for s in a_corpus:
            for w in s:
                if w in self.all_dictionary.word2idx:
                    self.a_idxs[ix] = self.all_dictionary.word2idx[w]
                    ix += 1

        tokens_b = 0


        ix = 0
        for s in b_corpus:
            for w in s:
                if w in self.all_dictionary.word2idx:
                    tokens_b += 1
                    self.b_dictionary.add_word(w)

        self.b_idxs = torch.zeros(tokens_b).long()
        for s in b_corpus:
            for w in s:
                if w in self.all_dictionary.word2idx:
                    self.b_idxs[ix] = self.all_dictionary.word2idx[w]
                    ix += 1



    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


    def dict_from_words(self, embs_name):
        self.dictionary = Dictionary_words()

        with open(embs_name, "r") as embs_file:
            dims = [int(x) for x in embs_file.readline().split(" ")]
            self.embeddings = torch.zeros(dims)
            ix = 0
            for f in embs_file:
                l = f.strip().split(" ")
                word, t = l[0], torch.FloatTensor([float(x) for x in l[1:]])
                self.dictionary.add_word(word)
                self.embeddings[ix] = t
                ix += 1
        return self.embeddings



def splitter(song):
    ix = song.find("K:")
    str_key = song[ix:].split('\n')[0]
    if ('m' in str_key or 'minor' in str_key):
        return True
    return False


if __name__ == "__main__":
    cm = Corpus_music("data/music/")
    path_maj = os.path.join("data/music/", 'major.abc')
    path_min = os.path.join("data/music/", 'minor.abc')

    f_maj = cm.get_frequencies(path_maj)
    cm.plot_histogram(f_maj, save_plot_path='plt.png')
    # print(f_maj)
    f_min = cm.get_frequencies(path_min)
    # print(f_min)
