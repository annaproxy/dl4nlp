#!/usr/bin/env python3

"""
Module for creating bag of word embeddings.
"""

import os.path
import argparse
import json
import itertools as it
import numpy as np

import torch

class Data():
    def __init__(self):
        data = self.read_file(args.in_dir + args.data)
        labels = self.read_file(args.in_dir + args.labels)
        data_batches = self.grouper(data, 100)
        labels_batches = self.grouper(labels, 100)

        self.data_batches = data_batches
        self.labels_batches = labels_batches
        self.vocabulary = self.read_file(args.v_dir + args.vocab)

        self.num_batches = len(self.data_batches)
        self.next_batch = 0

        languages = list(set(labels))
        languages.sort()

        self.lang_to_idx = {l:i for i, l in enumerate(languages)}
        self.idx_to_lang = {i:l for i, l in enumerate(languages)}

    @staticmethod
    def read_file(file_name):
        if not os.path.isfile(file_name):
            raise ValueError("Given input file cannot be found.")

        file = open(file_name, "r")
        if file_name.lower().endswith((".json")):
            contents = list(json.load(file))
        else:
            contents = [line.replace("\n", "") for line in file.readlines()]
        file.close()
        return contents

    @staticmethod
    def grouper(data, n, fillvalue=None):
        """ Create batches of data.

        Parameters
        ----------
        inputs : list
            list to be split up into batches
        n : int
            batch size
        fillvalue : object
            value used to fill last batch if not enough elements

        Returns
        -------
        list
            a list of batches
        """
        iters = [iter(data)] * n
        return list(it.zip_longest(*iters, fillvalue=fillvalue))

    def bag_of_words(self, data):
        """ Create character level bag of words embedding.

        Parameters
        ----------
        data : list
            batch of data to be embedded

        Returns
        -------
        np ndarray, lists how many times each character occurs in each paragraph

        """
        embeddings = np.ndarray((len(data), len(self.vocabulary)))
        for i, paragraph in enumerate(data):
            all_chars_paragraph = [char for char in paragraph]
            for j, char in enumerate(self.vocabulary):
                char_count = all_chars_paragraph.count(char)
                embeddings[i, j] = char_count
        return embeddings

    def get_next_embedded_batch(self):
        batch = self.data_batches[self.next_batch]
        self.next_batch += 1
        return self.bag_of_words(batch)


class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(len(self.vocabulary), 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

def main():
    raise NotImplementedError()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dir", default="./data/wili-2018/",
                        help="directory of input file")
    parser.add_argument("-data", default="x_train_sub.txt",
                        help="input file containing train data")
    parser.add_argument("-labels", default="y_train_sub.txt",
                        help="input file containing labels for train data")
    parser.add_argument("-v_dir", default="./data/vocabs/",
                        help="directory of vocab file")
    parser.add_argument("-vocab", default="full_vocab.json",
                        help="file vocabulary will be loaded from")
    parser.add_argument("-out_dir", default="./data/embeddings/",
                        help="directory output is saved to")
    parser.add_argument("-output", default="paragraph_embeddings.pickle",
                        help="name of output file")
    args = parser.parse_args()

    main()
