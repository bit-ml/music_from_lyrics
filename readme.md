# Music from Lyrics

This project is an attempt at generating music using recursive neural
networks and some NLP methods.

## Main architecture

The crux of the project represents a RNN - type neural network which is trained to associate musical patterns with specific sets of musical lyrics. The finality is the generation of music, in a minor or major key, depending on the nature of the lyrics fed to model.

## Context
The model also takes into account a previous musical "context", of varying length. In a musical context it is important to take into account previous musical notes in order to generate a good melodic continuation. 

## How to run
First of all we train the model by issuing
`python main.py --ncontext x ` 
Additionally being able to specify whether we should use `cuda` or not, and specify the files where we should save our model. The `x` accounts for the length of the previous context to take into consideration in our model prediction.

Afterwards we may issue `python generate_on_the_fly.py --ncontext x --checkpoint <model_name> --lyrics_path <file_with_lyrics>`. `--checkpoint` accounts for the model we choose to use in generation, and `--lyrics_path` accounts for the lyrics file which shall give the words for the melody generation. 

## Dataset
All necessary data apart from the pretrained glove embeddings come included in the repo. We recommend retrieving a set of embeddings from [here](https://github.com/3Top/word2vec-api#where-to-get-a-pretrained-models), then running `python quick_save.py <downloaded glove embeddings> <where_to_save_vocab> <where_to_save_embeddings>`. Afterwards run `main.py` with `--pretrained_vocab <saved vocab file>` and `--pretrained_embs <saved embs file>`.
Abc files come from http://www.atrilcoral.com, lyrics crawled from http://lyrics.wikia.com/.



## Credits
Main inspiration taken from https://github.com/pytorch/examples/tree/master/word_language_model
