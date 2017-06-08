# Bitdefender 2017

import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import data_words_music
import data
import model
import pdb
import sys

def save_to_file(model, char_len, fname):
   # Turn on evaluation mode which disables dropout.
   with open(fname, 'w') as f:
      model.eval()

      hidden  = model.init_hidden(1)
      ntokens = len(corpus_music.dictionary)
      input = Variable(torch.rand(2, 1).mul(ntokens).long(), volatile=True)
      train_data_lyrics = None
      train_data_music = None

      # again random selection between datasets
      data_choice = randint(0,1)
      if data_choice == 0:
         train_data_lyrics = train_data_lyrics_a
         train_data_music  = train_data_music_w
      else:
         train_data_lyrics = train_data_lyrics_b
         train_data_music  = train_data_music_j

      if args.cuda:
         input.data = input.data.cuda()

      last_chars = None

      if args.ncontext:
         if args.cuda:
            last_chars = [torch.autograd.Variable(torch.LongTensor(1,1).cuda().fill_(0)) \
                     for i in range(args.ncontext)]
         else:
            last_chars = [torch.autograd.Variable(torch.LongTensor(1,1).fill_(0)) \
                     for i in range(args.ncontext)]


      for i in range(char_len):
         input[1,:] = train_data_lyrics[i % len(train_data_lyrics),0]
         output, hidden = model(input, hidden, last_chars)
         char_weights = output.squeeze().data.div(args.temperature).exp().cpu()
         char_idx = torch.multinomial(char_weights, 1)[0]

         if args.ncontext:
            for n in range(args.ncontext):
                if n == 0:
                    last_chars[n] = torch.autograd.Variable(\
                            torch.LongTensor(1,1).fill_(char_idx))
                    if args.cuda:
                            last_chars[n] = last_chars[n].cuda()

                else:
                    last_chars[n] = last_chars[n - 1]


         input.data.fill_(char_idx)
         char = corpus_music.dictionary.idx2char[char_idx]
         f.write(char)
      f.write('\n')

parser = argparse.ArgumentParser(\
        description='PyTorch Music From lyrics generator')
parser.add_argument('--data', type=str, default='./data',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')

parser.add_argument('--save_model_directory', type=str,  default='/data/music_from_lyrics_data/models/',
                    help='path to save the final model')

parser.add_argument('--save_model_name', type=str,  default='model_context_27_mai_nc5nosen',
                    help='path to save the final model')

parser.add_argument('--save_every', type=int,  default=10,
                    help='number of epochs to run before saving model')
parser.add_argument('--optimizer', type=str,  default='adam',
                    help='optimizer to use for training')
parser.add_argument('--eval_chars', type=int, default='100',
                    help='number of chars to generate')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')

parser.add_argument('--pretrained_vocab', type=str, \
                    default="preloaded/vocab.bin",
                    help='path to pretrained vocabulary')

parser.add_argument('--pretrained_embs', type=str, default="preloaded/embs.bin",
                    help='path to pretrained embedddings')

parser.add_argument('--ncontext', type=int, default=None,
                    help='number of previous melody symbols to take into account')

parser.add_argument('--loss_directory', type=str, default="/data/music_from_lyrics_data/losses",
                    help='name of the loss folder')

parser.add_argument('--loss_name', type=str, default="loss_gru_2",
                    help='name of the loss file prefix')

parser.add_argument('--compress', type=bool, default=False,
                help='compress context embeddings')

parser.add_argument('--add_sentiment', type=bool, default=False,
                help='adds a sentiment value as input to the rnn')

args = parser.parse_args()

if args.cuda:
    torch.cuda.set_device(0)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
corpus_music    = data_words_music.Corpus_music(args.data + "/music")
corpus_lyrics   = data_words_music.Corpus_words(args.data + "/words",\
        args.pretrained_vocab, args.pretrained_embs)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = 10
train_data_music_j  = batchify(corpus_music.train_j, args.batch_size)
train_data_music_w  = batchify(corpus_music.train_w, args.batch_size)

train_data_lyrics_a  = batchify(corpus_lyrics.a_idxs, args.batch_size)
train_data_lyrics_b  = batchify(corpus_lyrics.b_idxs, args.batch_size)

train_data_sentiment_a  = batchify(corpus_lyrics.a_sentiments, args.batch_size)
train_data_sentiment_b  = batchify(corpus_lyrics.b_sentiments, args.batch_size)



###############################################################################
# Build the model
###############################################################################


ntokens_music   = len(corpus_music.dictionary)
ntokens_lyrics  = len(corpus_lyrics.all_dictionary)
model = model.RNNModel(args.model, ntokens_music, ntokens_lyrics , args.emsize, \
        args.nhid, args.nlayers, args.dropout, \
        args.tied,corpus_lyrics.embeddings, args.ncontext, args.compress, add_sentiment=args.add_sentiment)

if args.cuda:
    model.cuda()

# this goes all the way down into thnn, cannot be rewritten to return batch only
criterion = nn.CrossEntropyLoss()

optimizer = None
if args.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_cycle_batch(source, i, seq_len,evaluation = False, target = False, \
      prev=None):
    inc = i % len(source)
    sf  = (i + seq_len) % len(source)
    if sf >= inc:
        if target:
            data = Variable(source[inc:sf].view(-1))
        else:
            data = Variable(source[inc:sf], volatile=evaluation)
    else:
        if target:
            data = Variable(source[inc:].view(-1))
            if sf > 0:
                data2 = Variable(source[:sf].view(-1))
                data  = torch.cat([data, data2])
        else:

            data = Variable(source[inc:], volatile=evaluation)
            if sf > 0:
                data2 = Variable(source[:sf], volatile=evaluation)
                data  = torch.cat([data, data2])

    return data

def get_batch_cycle(source, source_lyrics, i, evaluation=False, source_sentiment=None):
    seq_len = args.bptt
    data    = get_cycle_batch(source, i, seq_len, evaluation)
    target  = get_cycle_batch(source, i + 1, seq_len, evaluation, True)
    data_lyrics    = get_cycle_batch(source_lyrics, i, seq_len, evaluation)
    data_sentiment = None
    if source_sentiment is not None:
        data_sentiment = get_cycle_batch(source_sentiment, i, \
                seq_len, evaluation)

    
    if source_sentiment is None:
        return data, target, data_lyrics
    else:
        return data, target, data_lyrics, data_sentiment

def evaluate():
    # Turn on evaluation mode which disables dropout.
    model.eval()

    hidden  = model.init_hidden(1)
    ntokens = len(corpus_music.dictionary)
    input = Variable(torch.rand(2, 1).mul(ntokens).long(), volatile=True)
    train_data_lyrics = None
    train_data_music = None
    train_data_sentiment = None

    # again random selection between datasets
    data_choice = randint(0,1)
    if data_choice == 0:
        train_data_lyrics = train_data_lyrics_a
        train_data_music  = train_data_music_w
        train_data_sentiment = train_data_sentiment_a 
    else:
        train_data_lyrics = train_data_lyrics_b
        train_data_music  = train_data_music_j
        train_data_sentiment = train_data_sentiment_b

    if args.cuda:
        input.data = input.data.cuda()

    last_chars = None

    if args.ncontext:
         if args.cuda:
            last_chars = [torch.autograd.Variable(torch.LongTensor(1,1)\
                  .cuda().fill_(0)) for i in range(args.ncontext)]
         else:
            last_chars = [torch.autograd.Variable(torch.LongTensor(1,1)\
                  .fill_(0)) for i in range(args.ncontext)]

    for i in range(args.eval_chars):
        input[1,:] = train_data_lyrics[i % len(train_data_lyrics),0]

        output, hidden = model(input, hidden, last_chars)

        char_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        char_idx = torch.multinomial(char_weights, 1)[0]
        input.data.fill_(char_idx)

        if args.ncontext:
            for n in range(args.ncontext):
                if n == 0:
                    last_chars[n] = torch.autograd.Variable(\
                            torch.LongTensor(1,1).fill_(char_idx))
                    if args.cuda:
                       last_chars[n] = last_chars[n].cuda()
                else:
                    last_chars[n] = last_chars[n - 1]
        char = corpus_music.dictionary.idx2char[char_idx]
        sys.stdout.write(char)
    sys.stdout.write('\n')
    sys.stdout.flush()
    save_to_file(model, 1000, "generated.abc")
    print("saved to file!")

from random import randint

import nltk
import nltk.sentiment

def get_last_chars(data_music, ncontext):
    if args.cuda:
      last_char = [torch.autograd.Variable(torch.LongTensor(35,20).cuda()\
            .fill_(0))\
            for i in range(ncontext)]
    else:
      last_char = [torch.autograd.Variable(torch.LongTensor(35,20).fill_(0))\
            for i in range(ncontext)]


    for j in range(ncontext):
        for i in range(data_music.size(1)):
            if i < (j + 1):
                continue
            last_char[j][i] = data_music[i - j - 1]
    return last_char


def train(ep_ix):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus_music.dictionary)
    hidden = model.init_hidden(args.batch_size)

    # we could treat alignment here
    max_len = max([train_data_music_w.size(0) - 1,\
            train_data_lyrics_a.size(0) - 1,
            train_data_music_j.size(0) - 1, train_data_lyrics_b.size(0)])

    losses = torch.Tensor(int(max_len / (args.bptt))).fill_(0)
    losses_ix = 0

    for batch, i in enumerate(range(0, max_len, args.bptt)):
        train_data_lyrics = None
        train_data_music = None
        train_data_sentiment = None
        # random choice between what dataset to use
        data_choice = randint(0,1)
        if data_choice == 0:
            train_data_lyrics = train_data_lyrics_a
            train_data_music  = train_data_music_w
            train_data_sentiment = train_data_sentiment_a
        else:
            train_data_lyrics = train_data_lyrics_b
            train_data_music  = train_data_music_j
            train_data_sentiment = train_data_sentiment_b

        data_music, targets_music, data_lyrics, data_sentiment = \
                get_batch_cycle(train_data_music, train_data_lyrics, i,\
                evaluation=False, source_sentiment=train_data_sentiment)

        last_chars = None
        if args.ncontext:
            last_chars = get_last_chars(data_music, args.ncontext)

        all_data = torch.cat((data_music, data_lyrics.long()))
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        if args.add_sentiment is False:
            data_sentiment = None
        output, hidden = model(all_data, hidden, last_chars, sentiment=data_sentiment)
        loss = criterion(output.view(-1, ntokens), targets_music)
        optimizer.zero_grad()
        loss.backward()

        if batch < len(losses):
            losses[batch] = loss.data[0]

        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            if losses_ix < losses.size(0):
                losses[losses_ix] = cur_loss
                losses_ix += 1
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, max_len // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

        # ln = args.loss_name + "_" + str(ep_ix) + ".pkl"
        # loss_path = os.path.join(args.loss_directory, ln)

        # with open(loss_path, 'wb') as f:
            # torch.save(losses, loss_path)
            # print("saved losses!")

        print("went through batch " + str(batch))

    return losses



# Loop over epochs.
lr = args.lr
best_val_loss = None

model_save_path = os.path.join(args.save_model_directory, args.save_model_name)

try:
    loss_means = torch.Tensor(args.epochs).fill_(0)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        losses = train(epoch)

        loss_mean = losses.mean()
        loss_means[epoch - 1] = loss_mean

        print(loss_means)
        evaluate()
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, \
                (time.time() - epoch_start_time)))
        print('-' * 89)

        if epoch % args.save_every == 0:
            suffix = str(epoch) + "_";

            with open(model_save_path + suffix + ".pt" , 'wb') as f:
                torch.save(model, f)

            with open(model_save_path + suffix + "_losses" , 'wb') as f:
                torch.save(losses, f)

            with open(model_save_path + suffix + "_losses_epoch" , 'wb') as f:
                torch.save(loss_means, f)


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
        # input.data.fill_(word_idx)
