import numpy as np
from pathlib import Path
import torch
from collections import defaultdict
import copy
import pickle

import tokenizers
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit, ByteLevel


class Tokenizer:

    def __init__(self, algo, vocab_size):

        pt = ByteLevel()
        self.alphabet = sorted(pt.alphabet())
        # Create a dictionary that maps characters to their position index in the list
        self.char_index_map = {char: str(index) for index, char in enumerate(self.alphabet)}

        self.algo = algo
        self.vocab_size = vocab_size
        if algo=='bpe':
            self.Trainer, Model = BpeTrainer, BPE
        elif algo=='wordpiece':
            self.Trainer, Model = WordPieceTrainer, WordPiece
        elif algo=='unigram':
            self.Trainer, Model = UnigramTrainer, Unigram
        else:
            raise NotImplementedError

        if algo=='wordpiece':
            self.tokenizer = tokenizers.Tokenizer(Model(unk_token="[UNK]", max_input_chars_per_word=100000))
            self.tokenizer.decoder = tokenizers.decoders.WordPiece()
        else:
            self.tokenizer = tokenizers.Tokenizer(Model())
        self.tokenizer.pre_tokenizer = WhitespaceSplit()

    def train(self, corpus, min_frequency, max_token_length, verbose=False):
        corpus = self.textualize(corpus)
        trainer = self.Trainer(vocab_size=self.vocab_size,
                               special_tokens=["[UNK]"],
                               min_frequency=min_frequency,
                               max_token_length=max_token_length)

        self.tokenizer.train_from_iterator([corpus], trainer=trainer)

        vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(vocab)
        if verbose:
            print("Learned vocab size: {}".format(len(vocab)))
            print("Max token length: {}".format(max([len(x) for x in vocab.keys()])))
            # print('vocab', vocab)

    def textualize(self, raw):
        assert type(raw) == list
        if type(raw[0])==int:
            raw = [raw]
        # raw is list of list of integers
        return ' '.join([self.to_alphabet(word) for word in raw])

    def to_alphabet(self, chars):
        # list of character ids in the alphabet
        return ''.join([self.alphabet[char] for char in chars])

    def detextualize(self, text):
        # alphabet to original ids
        text = ' '.join(text.replace(' ','')) # add white space between characters
        decoded_text = ''.join([self.char_index_map[char] if char in self.char_index_map else char for char in text])
        return [ int(i) for i in decoded_text.split(' ')]

    def encode(self, raw, verbose=False):
        """ Encode a list of original ids to a list of token ids. """
        corpus = self.textualize(raw)
        encoded_text = self.tokenizer.encode(corpus)
        if verbose:
            print('raw:', raw)
            print('alphabet:', corpus)
            print('encoded token:', encoded_text.tokens)
            print('# raw ids:', len(raw))
            print('# tokens:', len(encoded_text.tokens))
            print('# unique tokens:', len(set(encoded_text.tokens)))
        return encoded_text.ids  # id of tokens

    def decode(self, token_ids, verbose=False):
        """ Decode a list of token ids to a list of original ids """
        decoded_text = self.tokenizer.decode(token_ids)  # this is at the alphabet level
        original = self.detextualize(decoded_text)

        if verbose:
            print('decoded (alphabet)', decoded_text)
            print('decoded (raw)', original)
        return original




def main(args):
    # Test
    import pickle
    data = pickle.load(open(args.file, 'rb'))[-1]

    for vocab_size in [10]:
        for min_frequency in [3,5,10,15]:
            for max_token_length in [5,10,15,20]:
                tokenizer = Tokenizer(algo=args.algo, vocab_size=vocab_size)
                tokenizer.train(data, min_frequency=min_frequency, max_token_length=max_token_length, verbose=False)
                with open('/mount_point/temporal_action_abstraction/tokenizers/tokenizer_code10_vocab{}_minfreq{}_maxlen{}.pkl'.format(vocab_size, min_frequency, max_token_length), 'wb') as f:
                     pickle.dump(tokenizer, f)
    
    #tokenizer = Tokenizer(algo=args.algo, vocab_size=args.vocab_size)
    #tokenizer.train(data, min_frequency=args.min_frequency, max_token_length=args.max_token_length, verbose=False)
    
    

    #print("Learned vocab size: {}".format(len(vocab)))
    
    
    # text = data[2]
    # # print(f'\nraw input: {text}\n# of codes {len(text)}')
    # print(text)
    # token_ids = tokenizer.encode(text, verbose=False)
    # print(token_ids, tokenizer.decode([20] , verbose=False))
    # print(tokenizer.decode(token_ids, verbose=False))
    ### A function that maps from a token_id to length
    ### A function that maps from a token_id to first code
    #token_ids = [tokenizer.encode(text[i:], verbose=False)[0] for i in range(len(text))]
    #print(text)
    #print(token_ids[0], tokenizer.decode([token_ids[0]], verbose=False))
    
    #reconstructed_text = tokenizer.decode(token_ids, verbose=False)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', default=5000, type=int)
    parser.add_argument('--min_frequency', default=5, type=int)
    parser.add_argument('--max_token_length', default=20, type=int)
    parser.add_argument('--file', default='/mount_point/temporal_action_abstraction/vocab_mt45_obs_dependent_code20.pkl', type=str)
    parser.add_argument('--algo', default='bpe', type=str)
    args = parser.parse_args()
    main(args)