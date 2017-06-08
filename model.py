# Bitdefender 2017

import torch.nn as nn
import torch
from torch.autograd import Variable
import sys
import pdb
import nltk
import nltk.sentiment.vader

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ntokens_lyrics, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, pre_embeds=None,
            ncontext=None, compressor=None, add_sentiment=False):
        super(RNNModel, self).__init__()
        self.drop           = nn.Dropout(dropout)
        self.encoder        = nn.Embedding(ntoken, ninp)
        self.encoder_lyrics = nn.Embedding(ntokens_lyrics, ninp)
        self.compressor     = None
        self.ncontext       = ncontext
        self.ninp           = ninp
        self.sentiment_evaluator = nltk.sentiment.vader.SentimentIntensityAnalyzer()

        if pre_embeds is not None:
            embeddings = pre_embeds #torch.from_numpy(pre_embeds.wv.syn0)
            if compressor and self.ncontext:
                self.compressor      = nn.Linear(ncontext * ninp, ninp)
            
            self.encoder_lyrics.weight = torch.nn.Parameter(embeddings)
        else:
            pass

        if rnn_type in ['LSTM', 'GRU']:
            added = ncontext if ncontext else 0
            a_s = 1 if add_sentiment is not False else 0 
            if self.compressor:
                added = 1

            self.rnn = getattr(nn, rnn_type)((2 + added) * ninp + a_s, nhid, nlayers)#, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity)#, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder_lyrics.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, extra_notes, sentiment=None):
        half = int(input.size(0) / 2)
        data        = input[0:half,:]
        data_lyrics = input[half:,:]
        emb = self.drop(self.encoder(data))
        emb_size = emb.size()
        emb2 = self.drop(self.encoder_lyrics(data_lyrics))
        emb = self.encoder(data)
        emb2 = self.encoder_lyrics(data_lyrics)
        emb_combined = torch.cat([emb, emb2],2)
        
        if not self.compressor and (self.ncontext is not None):
            for i in range(self.ncontext):
                emb_extra = self.drop(self.encoder(extra_notes[i]))
                emb_combined = torch.cat([emb_combined, emb_extra], 2)

        elif self.compressor and (self.ncontext is not None):

            embs_ = None#torch.autograd.Variable(torch.FloatTensor().cuda())
            for i in range(self.ncontext):
                emb_extra = self.drop(self.encoder(extra_notes[i]))
                if embs_ is None:
                    embs_ = emb_extra
                else:
                    embs_ = torch.cat([embs_, emb_extra], 2)

            embs_size = embs_.size()
            embs_     = embs_.view(-1, self.ncontext * self.ninp)
            embs_compressed = self.compressor(embs_)
            embs_compressed = embs_compressed.view(emb_size)

            emb_combined = torch.cat([emb_combined, embs_compressed], 2)

        if sentiment is not None: # account for sentiment as well
            sentiment = sentiment.unsqueeze(2)
            emb_combined = torch.cat([emb_combined, sentiment], 2)

        output, hidden = self.rnn(emb_combined, hidden)
        output = self.drop(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1),\
                output.size(2)))
        return decoded.view(output.size(0), output.size(1),\
                decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
