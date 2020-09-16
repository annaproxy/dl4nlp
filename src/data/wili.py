import os, sys

import random
import torch
import numpy  as np
from torch.utils.data import Dataset
import torch.utils.data as data
from torchvision import datasets
import json


class WiliDataLoader(Dataset):
    def __init__(self, data_path, label_path, sequence_length=32, n_slices=8):

        self.data_path = data_path
        self.label_path = label_path
        self.sequence_length = sequence_length
                
        self.lines, self.line_languages = self.load_lines()
        
        self.languages = sorted(list(set(self.line_languages)))
        self.languages.sort()
        
        self.lang_to_idx = { l:i for i,l in enumerate(self.languages) }
        self.idx_to_lang = { i:l for i,l in enumerate(self.languages) }        
        
        self.vocab_dict = json.load(open('./data/vocabs/full_vocab.json'))
        self.vocab_list = [key for key in self.vocab_dict]
        self.vocab_size = len(self.vocab_list)
        
        print("Vocabulary of size: {}".format(self.vocab_size))
        
        self.char_to_idx = { ch:i for i,ch in enumerate(self.vocab_list) }
        self.idx_to_char = { i:ch for i,ch in enumerate(self.vocab_list) }
        

    def load_lines(self):
        """
        Each line is a list of characters belonging to a specific language.
        Skipping the "/n" token.
        """
        
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
        lines = [list(paragraph)[:-1] for paragraph in lines]
        # lines = bpe shit for paragraph in lines
        
        with open(self.label_path, 'r') as f:
            languages = f.readlines()
        languages = [language[:-1] for language in languages]
        
        print('Loaded language paragraphs from: %s (%d)' % (self.data_path, len(lines)))

        return lines, languages


    def __getitem__(self, index):
        """
        Get the paragraph (line) and corresponding language.
        Get a random sequence of sequence length within that paragraph.
        """
        paragraph, language = self.lines[index], self.line_languages[index]
        paragraph_length = len(paragraph)
        
        offset = np.random.randint(0, paragraph_length-self.sequence_length)
        inputs = np.array([self.char_to_idx[ch] for ch in paragraph[offset:offset+self.sequence_length]])
        target = self.lang_to_idx[language]
        
        return inputs, target
        

    def __len__(self):
        return len(self.lines)


class WiliBytesDataLoader(Dataset):
    def __init__(self, data_path, label_path, sequence_length=30, n_slices=8):

        self.data_path = data_path
        self.label_path = label_path
        self.sequence_length = sequence_length
                
        self.lines, self.line_languages = self.load_lines()
        
        self.languages = sorted(list(set(self.line_languages)))
        self.languages.sort()
        
        self.lang_to_idx = { l:i for i,l in enumerate(self.languages) }
        self.idx_to_lang = { i:l for i,l in enumerate(self.languages) }        
        
        self.vocab_dict = json.load(open('./data/vocabs/full_bytes_vocab.json'))
        self.vocab_list = [key for key in self.vocab_dict]
        self.vocab_size = len(self.vocab_list)
        
        print("Vocabulary of size: {}".format(self.vocab_size))
        

    def load_lines(self):
        """
        Each line is a list of integers that represent subwords
        """
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
        lines = [[int(ch) for ch in paragraph.split()] for paragraph in lines]
        
        with open(self.label_path, 'r') as f:
            languages = f.readlines()
        languages = [language[:-1] for language in languages]
        
        print('Loaded language paragraphs from: %s (%d)' % (self.data_path, len(lines)))

        return lines, languages


    def __getitem__(self, index):
        """
        Get the paragraph (line) and corresponding language.
        Get a random sequence of sequence length within that paragraph.
        """
        paragraph, language = self.lines[index], self.line_languages[index]
        paragraph_length = len(paragraph)
        
        offset = np.random.randint(0, paragraph_length-self.sequence_length)
        inputs = np.array([ch for ch in paragraph[offset:offset+self.sequence_length]])
        target = self.lang_to_idx[language]
        
        return inputs, target
        

    def __len__(self):
        return len(self.lines)