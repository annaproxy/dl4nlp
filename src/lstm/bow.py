#!/usr/bin/env python3

"""
Module for creating bag of word embeddings.
"""
import os.path
import argparse
import json
import itertools as it
import numpy as np
import random as rd
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import classification_report

import torch
import torch.nn as nn

# import pdb; pdb.set_trace()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Data():
    def __init__(self):
        data = self.read_file(args.in_dir + args.data)
        labels = self.read_file(args.in_dir + args.labels)

        fixed_seed = rd.random()

        rd.Random(fixed_seed).shuffle(data)
        rd.Random(fixed_seed).shuffle(labels)

        data_batches = self.grouper(data, args.batch_size)
        labels_batches = self.grouper(labels, args.batch_size)

        if len(data) % args.batch_size != 0:
            data_batches[-1] = [paragraph for paragraph in data_batches[-1]
                                if paragraph is not None]
            labels_batches[-1] = [label for label in labels_batches[-1]
                                  if label is not None]

        self.data_batches = data_batches
        self.labels_batches = labels_batches

        self.vocabulary = self.read_file(args.v_dir + args.vocab)
        self.char_to_idx = {c:i for i, c in enumerate(self.vocabulary)}
        self.idx_to_char = {i:c for i, c in enumerate(self.vocabulary)}

        self.num_batches = len(self.data_batches)
        self.next_batch = 0

        languages = list(set(labels))
        languages.sort()

        self.languages = languages
        self.lang_to_idx = {l:i for i, l in enumerate(languages)}
        self.idx_to_lang = {i:l for i, l in enumerate(languages)}

    def reset_next_batch(self):
        self.next_batch = 0

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
    def grouper(data, n):
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
        return list(it.zip_longest(*iters))

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
        embeddings = np.zeros((len(data), len(self.vocabulary)))
        for i, paragraph in enumerate(data):
            all_chars_paragraph = np.array([char for char in paragraph])
            uniques, counts = np.unique(all_chars_paragraph, return_counts=True)
            uniques = [self.char_to_idx[unique] for unique in uniques]
            counts = counts.astype(np.int)
            embeddings[i, uniques] = counts
        return embeddings

    def get_next_embedded_batch(self):
        batch = self.data_batches[self.next_batch]
        labels0 = self.labels_batches[self.next_batch]
        labels = torch.zeros(len(labels0))
        for i, label in enumerate(labels0):
            labels[i] = self.lang_to_idx[label]

        self.next_batch += 1
        return self.bag_of_words(batch), labels.long()

class KL(nn.Module):
    def __init__(self, divisor=2):
        super().__init__()
        self.c1 = 1.16145124
        self.c2 = -1.50204118
        self.c3 = 0.58629921
        self.divisor = divisor

    def forward(self, alpha):
        return (0.5*alpha.log() + self.c1*alpha + self.c2*alpha**2 + self.c3*alpha**3) / self.divisor

class BayesianDropout(nn.Module):
    """
    We follow Max and Kingma paper "Variational Dropout and the Local Reparameterization Trick" (2015)
    Here alpha should be p / (1-p), as shown in the paper.

    """
    def __init__(self, dropout_rate, weight_dim, cuda=True):
        super().__init__()
        self.weight_dim = weight_dim
        alpha = dropout_rate / (1 - dropout_rate)
        self.alpha = alpha
        self.cuda = cuda
        self._alpha = nn.Parameter(torch.ones(weight_dim)*alpha)

    def forward(self, x):
        # Unit Gaussian prior
        noise = torch.autograd.Variable(torch.randn(x.shape)).to(torch.device('cuda' if self.cuda else "cpu"))

        # max=1.0 corresponds with a dropout rate of 0.5 (section 3.3)
        self._alpha.data = torch.clamp(self._alpha.data, max=1.0)
        noise *= self._alpha
        return x * noise

class LinearRegressionModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

class ExtraLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ExtraLinear, self).__init__()
        self.linear1 = nn.Linear(in_features, 512)
        # self.dropout = BayesianDropout(0.5, 512)
        self.non_linear = nn.ReLU()
        self.linear2 = nn.Linear(512, out_features)

    def forward(self, x):
        linear1 = self.linear1(x)
        # linear1 = self.dropout(linear1)
        non_linear = self.non_linear(linear1)
        y_pred = self.linear2(non_linear)
        return y_pred

def train(data, model):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters())
    # kl = KL(divisor=50)
    for epoch in range(100):
        running_loss = 0.0
        for i in tqdm(range(data.num_batches)):
            batch, labels = data.get_next_embedded_batch()
            batch = torch.from_numpy(batch).float().to(device)
            labels = labels.to(device)

            prediction = model.forward(batch)
            # _alpha = model.dropout._alpha
            # kl_divergence = torch.sum(kl(_alpha))
            # print("KL:",kl_divergence.item())

            optimizer.zero_grad()
            loss = criterion(prediction, labels)
            # print("CROSS LOSS: ", cross.item())
            # loss =  cross + kl_divergence
            loss.backward()
            optimizer.step()
            # print('epoch {}, it {}, loss {}'.format(epoch, i, loss.item()))

            running_loss += loss.item()
            if (i + 1) % 300 == 0:
                print('[epoch %f, iteration %5d] loss: %.3f' %
                      (int(epoch) + 1, i + 1, running_loss / 300))
                running_loss = 0.0
        torch.save(model.state_dict(), args.m_dir + args.output + ".pt")
        data.reset_next_batch()

def test(data, model):
    model.eval()
    all_predictions = []
    all_labels = []
    for _ in tqdm(range(data.num_batches)):
        batch, labels = data.get_next_embedded_batch()
        batch = torch.from_numpy(batch).float().to(device)
        labels = labels.to(device)

        prediction = model.forward(batch)
        prediction = torch.argmax(prediction, 1)

        all_predictions.extend(prediction.tolist())
        all_labels.extend(labels.tolist())
    target_names = data.languages
    return classification_report(all_labels, all_predictions,
                                 target_names=target_names, output_dict=True)

def main():
    data = Data()

    if args.model == "linreg":
        model = LinearRegressionModel(len(data.vocabulary), len(data.languages))
    elif args.model == "hidlay":
        model = ExtraLinear(len(data.vocabulary), len(data.languages))
    else:
        raise ValueError("Not sure what model you want to use.")
    model = model.to(device)

    if args.mode == "train":
        train(data, model)
    elif args.mode == "test":
        model.load_state_dict(torch.load(args.m_dir + args.output + ".pt"))
        report = test(data, model)
        df = pd.DataFrame(report).transpose()
        df.to_csv("classification_report_" + args.output + ".csv", index=True)
    else:
        raise ValueError("Invalid value for argument mode")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dir", default="./data/wili-2018/",
                        help="directory of input file")
    parser.add_argument("-data", default="x_train_sub_clean.txt",
                        help="input file containing train data")
    parser.add_argument("-labels", default="y_train_sub_clean.txt",
                        help="input file containing labels for train data")
    parser.add_argument("-v_dir", default="./data/vocabs/",
                        help="directory of vocab file")
    parser.add_argument("-vocab", default="full_vocab.json",
                        help="file vocabulary will be loaded from")
    parser.add_argument("-m_dir", default="../models/linear/",
                        help="directory the model is saved in")
    parser.add_argument("-output", default="hidden_layer_model_clean",
                        help="file model is saved to")
    parser.add_argument("-mode", default="train",
                        help="train or test")
    parser.add_argument("-model", default="hidlay",
                        help="'linreg' for linear regression or 'hidlay' for hidden layer")
    parser.add_argument("-batch_size", type=int, default=100,
                        help="... you really need help with this one?")
    args = parser.parse_args()

    main()
