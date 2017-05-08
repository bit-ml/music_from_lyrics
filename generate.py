###############################################################################
# Bitdefender 2017
#
# Pytorch music from lyrics
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable
#from main import batchify
import data_words_music
import pdb
parser = argparse.ArgumentParser(description='PyTorch Music generation')

# Model parameters.
parser.add_argument('--data', type=str, default='./data',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--A', type=int, default=0,
                    help='lyrics choice')

args = parser.parse_args()


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

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()



corpus_lyrics   = data_words_music.Corpus_words(args.data + "/words")
corpus_music = data_words_music.Corpus_music(args.data + "/music")

if args.A == 0:
   test_data_lyrics    = batchify(corpus_lyrics.a_idxs, 1)
else:
   test_data_lyrics    = batchify(corpus_lyrics.b_idxs, 1)






ntokens = len(corpus_music.dictionary)
hidden = model.init_hidden(1)
input = Variable(torch.rand(2, 1).mul(ntokens).long().cuda(), volatile=True)


for index in range(1): # how many?

    with open( args.outf + '_lyrics_' + str(index) , 'w') as outf_lyrics:
        with open(args.outf + '_mel_'+ str(index) , 'w') as outf:
            if args.A == 0:
                outf.write('X: trist\n')
                outf.write('K: Am\n')
            else:
                outf.write('X: vesel\n')
                outf.write('K: C\n')
            for i in range(args.words):
                word_idx = test_data_lyrics[i + index * 2000,0]
                input[1,:] = word_idx
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
                note_idx = torch.multinomial(word_weights, 1)[0]
        
                input.data.fill_(note_idx)
                char = corpus_music.dictionary.idx2char[note_idx]
                
                word = corpus_lyrics.all_dictionary.idx2word[word_idx]

                outf.write(char)
                outf_lyrics.write(word + " ");
                if i % args.log_interval == 0:
                    print('| Generated {}/{} words'.format(i, args.words))
