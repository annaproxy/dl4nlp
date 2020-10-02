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
import torch.optim as optim
import torch.nn.functional as f

# import pdb; pdb.set_trace()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Data():
    def __init__(self):
        self.data = self.read_file(args.in_dir + args.data)
        self.labels = self.read_file(args.in_dir + args.labels)

        fixed_seed = rd.random()

        rd.Random(fixed_seed).shuffle(self.data)
        rd.Random(fixed_seed).shuffle(self.labels)

        data_batches = self.grouper(self.data, args.batch_size)
        labels_batches = self.grouper(self.labels, args.batch_size)

        if len(self.data) % args.batch_size != 0:
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

        languages = list(set(self.labels))
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

    def forward(self, log_alpha):
        alpha = torch.exp(log_alpha)
        divergence = -(0.5*log_alpha + self.c1*alpha + self.c2*alpha**2 + self.c3*alpha**3) / self.divisor
        return divergence.mean()

class BayesianDropout(nn.Module):
    """
    We follow Max and Kingma paper "Variational Dropout and the Local Reparameterization Trick" (2015)
    Here alpha should be p / (1-p), as shown in the paper.
    """
    def __init__(self, dropout_rate, weight_dim, cuda=True):
        super().__init__()
        self.alpha = dropout_rate / (1 - dropout_rate)
        self._log_alpha = nn.Parameter(torch.log(torch.ones(weight_dim)*self.alpha), requires_grad=True)

        # init log_alpha params
        sd = 1. / torch.sqrt(torch.tensor(weight_dim).float())
        self._log_alpha.data.uniform_(-sd, sd)

    def forward(self, x, eval=False):
        # Gaussian prior
        if eval:
            return x
        noise = torch.randn(x.shape).to(torch.device('cuda' if self.cuda else "cpu")) + 1

        # max=1.0 corresponds with a dropout rate of 0.5 (section 3.3)
        # self._log_alpha.data = torch.clamp(self._log_alpha.data, max=1.0)
        self._log_alpha.data = self._log_alpha.data.masked_fill(self._log_alpha.data > 1.0, 0)
        self._log_alpha.data = self._log_alpha.data.masked_fill(self._log_alpha.data < -1.0, 0)

        noise *= torch.exp(self._log_alpha)
        return x * noise

class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = BayesianDropout(0.5, out_features)

    def forward(self, x, eval=False):
        linear = self.linear(x)
        prediction = self.dropout(linear, eval)
        return prediction

class ExtraLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ExtraLinear, self).__init__()
        self.linear1 = nn.Linear(in_features, 512)
        self.dropout = BayesianDropout(0.5, 512)
        self.non_linear = nn.ReLU()
        self.linear2 = nn.Linear(512, out_features)

    def forward(self, x, eval=False):
        linear1 = self.linear1(x)
        dropout = self.dropout(linear1, eval)
        non_linear = self.non_linear(dropout)
        prediction = self.linear2(non_linear)
        return prediction

def train(data, model):
    model.train()
    criterion = nn.CrossEntropyLoss()
    kl = KL(divisor=2)
    optimizer = optim.Adam(params=model.parameters())
    for epoch in range(100):
        running_loss = 0.0
        running_kl = 0.0
        for i in tqdm(range(data.num_batches)):
            batch, labels = data.get_next_embedded_batch()
            batch = torch.from_numpy(batch).float().to(device)
            labels = labels.to(device)

            prediction = model.forward(batch)
            log_alpha = model.dropout._log_alpha
            kl_divergence = kl(log_alpha)

            # if args.model == "linreg":
            #     log_alpha = model.dropout._log_alpha
            #     kl_divergence = kl(log_alpha)
            # else:
            #     log_alpha1 = model.dropout1._log_alpha
            #     log_alpha2 = model.dropout2._log_alpha
            #     kl_divergence = kl(log_alpha1) + kl(log_alpha2)

            optimizer.zero_grad()
            cross_entropy = criterion(prediction, labels)
            loss = cross_entropy + kl_divergence
            loss.backward()
            optimizer.step()
            # print('epoch {}, it {}, loss {}, kl {}'.format(epoch, i, loss.item(), kl_divergence))

            running_loss += loss.item()
            running_kl += kl_divergence
            if (i + 1) % 300 == 0:
                print('[epoch %f, iteration %5d] loss: %.3f, kl: %.3f' %
                      (int(epoch) + 1, i + 1, running_loss / 300, running_kl / 300))
                running_loss = 0.0
        torch.save(model.state_dict(), args.m_dir + args.output + ".pt")
        data.reset_next_batch()

# def test(data, model):
#     model.eval()
#     all_predictions = []
#     all_labels = []
#     for _ in tqdm(range(data.num_batches)):
#         batch, labels = data.get_next_embedded_batch()
#         batch = torch.from_numpy(batch).float().to(device)
#         labels = labels.to(device)
#
#         prediction = model.forward(batch, eval=False)
#         prediction = torch.argmax(prediction, 1)
#
#         all_predictions.extend(prediction.tolist())
#         all_labels.extend(labels.tolist())
#     target_names = data.languages
#     report = classification_report(all_labels, all_predictions,
#                                  target_names=target_names, output_dict=True)
#     df = pd.DataFrame(report).transpose()
#     df.to_csv("classification_report_" + args.output + ".csv", index=True)

def test_uncertainty(data, model):
    model.eval()
    y_pred =[]
    y_true = []
    accuracies = []
    for i, datapoint in enumerate(tqdm(data.data)):
        datapoint = data.bag_of_words([datapoint])
        datapoint = torch.from_numpy(datapoint).float().to(device)
        datapoint_probs = torch.zeros(50, len(data.languages))
        for j in range(50):
            logits = model(datapoint)
            probabilities = f.softmax(logits, dim=1)
            datapoint_probs[j, :] = probabilities
        standard_deviations = torch.std(datapoint_probs, dim=0)
        means = torch.mean(datapoint_probs, dim=0)
        prediction = torch.argmax(means).item()

        label = data.lang_to_idx[data.labels[i]]

        y_pred.append(prediction)
        y_true.append(label)

        correct = (prediction == label)
        accuracies.append(correct)

        means = [round(mean, 6) for mean in means.cpu().detach().numpy()]
        std = [round(std_i, 6) for std_i in standard_deviations.cpu().detach().numpy()]

        with open(args.output + ".csv", "a") as file:
            file.write(str(i)+"; "+ data.idx_to_lang[prediction]+"; " + \
                        data.idx_to_lang[label]+"; "+str(means)+"; " + \
                        str(std)+"\n")

    accuracy = round(np.sum(accuracies)/(len(data.data)),4)
    print("Accuracy: ", accuracy)
    target_names = data.languages
    df = pd.DataFrame(classification_report(y_true, y_pred,
                                            target_names=target_names,
                                            output_dict=True)).transpose()
    df.to_csv("classification_report_uncertainty_{}.csv".format(args.output), index=True)

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
        # test(data, model)
        test_uncertainty(data, model)
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
