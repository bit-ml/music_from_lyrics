# Bitdefender 2017

import os
import torch

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
        self.exclusion_dict = set(["X:", "T:", "%", "K:", "P:", "M:", "S:"])
        self.train = self.tokenize(os.path.join(path, 'jigs.txt'))


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


class Corpus_words(object):
    def __init__(self, path):
        self.dictionary = Dictionary_words()
        self.train = self.tokenize(os.path.join(path, 'metallica.txt'))

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
