################################################################################
# Bitdefender 2017
#
# Pytorch music from lyrics
#
################################################################################
import argparse
import torch
from torch.autograd import Variable
import data_words_music
import pdb
import sys
import re
parser = argparse.ArgumentParser(description='PyTorch Music generation')

# Model parameters.
parser.add_argument('--data', type=str, default='./data',
      help='location of the data corpus')

parser.add_argument('--checkpoint', type=str, default='./models/model_context30_.pt',
      help='model checkpoint to use')

parser.add_argument('--outd', type=str, default='generated_music',
      help='output directory for generated music')

parser.add_argument('--outf', type=str, default='context_music',
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

parser.add_argument('--dchoice', type=int, default=0,
      help='lyrics choice')

parser.add_argument('--lyrics_path', type=str, default=None,
      help='path of lyrics used for music generation')

parser.add_argument('--use_glove', type=bool, default=True,
      help='should use pretrained glove embeddings for text')

parser.add_argument('--pretrained_vocab', type=str, default="preloaded/vocab.bin",
      help='path to pretrained vocabulary')

parser.add_argument('--pretrained_embs', type=str, default="preloaded/embs.bin",
      help='path to pretrained embedddings')

parser.add_argument('--ncontext', type=int, default=None,
      help='number of previous melody symbols to take into account')

parser.add_argument('--num_songs', type=int, default=10,
      help="number of melodies to generate")

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

gen_lyrics = None

if args.lyrics_path:
   with open(args.lyrics_path, "r") as inp:
      gen_lyrics = inp.read()

print(gen_lyrics)
gen_lyrics = re.sub(r"[^A-Za-z]", " ", gen_lyrics.strip()).split(" ")
gen_lyrics = [x.lower() for x in gen_lyrics if len(x) > 0]

corpus_lyrics = None
if args.use_glove:
   corpus_lyrics   = data_words_music.Corpus_words(args.data + "/words",\
         args.pretrained_vocab, args.pretrained_embs)
else:
   corpus_lyrics   = data_words_music.Corpus_words(args.data + "/words")

corpus_music = data_words_music.Corpus_music(args.data + "/music")


if args.dchoice == 0:
   test_data_lyrics    = batchify(corpus_lyrics.a_idxs, 1)
else:
   test_data_lyrics    = batchify(corpus_lyrics.b_idxs, 1)


ntokens = len(corpus_music.dictionary)
hidden = model.init_hidden(1)
input = Variable(torch.rand(2, 1).mul(ntokens).long().cuda(), volatile=True)

import os.path

save_folder = os.path.join(args.outd, args.outf)

num_melodies = args.num_melodies if not args.lyrics_path else 1

for index in range(num_melodies): # how many?
   if args.ncontext:
      if args.cuda:
         last_chars = [torch.autograd.Variable(torch.LongTensor(1,1)\
               .cuda().fill_(0)) for i in range(args.ncontext)]
      else:
         last_chars = [torch.autograd.Variable(torch.LongTensor(1,1)\
               .fill_(0)) for i in range(args.ncontext)]

      with open(save_folder + '_lyrics_' + str(index) , 'w') as outf_lyrics:
         with open(save_folder + '_mel_'+ str(index) , 'w') as outf_song:
            print(save_folder + '_lyrics_' + str(index))
            if args.dchoice == 0:
               outf_song.write('X: trist\n')
               outf_song.write('K: Am\n')
            else:
               outf_song.write('X: vesel\n')
               outf_song.write('K: C\n')

            iterator = enumerate(gen_lyrics) if gen_lyrics is not None\
               else enumerate(range(args.word))

            for i,w in iterator:
               print(i)
               word_idx = 0
               if gen_lyrics:
                  if w in corpus_lyrics.all_dictionary.word2idx:
                     word_idx = corpus_lyrics.all_dictionary.word2idx[w]
                  else:
                     print(w + " not found!")
               else:
                   word_idx = test_data_lyrics[i + index * 2000,0]

               input[1,:] = word_idx
               output, hidden = model(input, hidden, last_chars)
               word_weights = output.squeeze().\
                     data.div(args.temperature).exp().cpu()
               note_idx = torch.multinomial(word_weights, 1)[0]

               if args.ncontext:
                  for n in range(args.ncontext):
                     if n == 0:
                        last_chars[n] = torch.autograd.Variable(\
                              torch.LongTensor(1,1).fill_(note_idx))
                        if args.cuda:
                           last_chars[n] = last_chars[n].cuda()
                     else:
                        last_chars[n] = last_chars[n - 1]

               input.data.fill_(note_idx)
               char = corpus_music.dictionary.idx2char[note_idx]
               word = corpus_lyrics.all_dictionary.idx2word[word_idx]

               outf_song.write(char)
               outf_lyrics.write(word + " ");
               if i + 1 % args.log_interval == 0:
                  print('| Generated {}/{} words'.format(i, args.words))
