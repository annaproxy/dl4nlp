
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.optim as optim
from data.load_data import get_wili_data, get_wili_data_bytes

import os
import sys
import torch
import random
import argparse
import numpy as np
from GPT2.model import (GPT2LMHeadModel)
from GPT2.config import GPT2Config
from utils.utils import *
from utils.config import config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, training_loader, validation_loader, validation_data, config):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    avg_train_loss = []
    train_loss = []
    accuracy = 0
    for epoch in range(config.epochs):
        print("Starting Epoch: {}".format(epoch))
        for i, (inputs, labels) in enumerate(training_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            output = model(inputs)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            if  (i+1) % 11 == 0:
                print('%d iterations' % (i+1))
                avg = np.mean(train_loss[-50:])
                avg_train_loss.append(avg)
                print('Loss: %.3f' % avg)
                print()
                #accuracy = validate_paragraphs(model, validation_data, validation_loader, save_classification_report=False)
                #print("Current Accuracy: {}, After {} iterations in Epoch {}".format(accuracy, i, epoch))
                #model.train()
                #torch.save(model.state_dict(), "./models/gpt/"+str(epoch)+"_"+str(accuracy)+".pt")
                #torch.save(model.state_dict(), "./models/gpt/"+str(epoch)+".pt")

        scheduler.step()
        torch.save(model.state_dict(), "./models/gpt/"+str(epoch)+"_"+str(accuracy)+".pt")
    #write_results((avg_train_loss, val_loss, val_accuracy), model_type+"_")
    print("Iterators Done")

def main():

    gpt_config = GPT2Config()
    param_config = config()

    model = GPT2LMHeadModel(gpt_config)

    #with open("./models/gpt/gpt2-pytorch_model.bin", 'rb') as f:
    #    state_dict = torch.load(f, map_location='cpu' if not torch.cuda.is_available() else None)
    #    print("GPT-2 Model Loaded.")

    #   model = load_weight(model, state_dict)
    if param_config.model_checkpoint is not None:
        with open(param_config.model_checkpoint, 'rb') as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)
            print("Model Loaded From: {}".format(param_config.model_checkpoint))
    model.to(device)

    # Load Data
    if param_config.input == 'bytes':
        # Load Data for bytes
        training_data, validation_data = get_wili_data_bytes(param_config)
    else:
        # Load Data
        training_data, validation_data = get_wili_data(param_config)


    training_loader = DataLoader(training_data,
                             batch_size=param_config.batch_size,
                             shuffle=True,
                             drop_last=False)

    validation_loader = DataLoader(validation_data,
                             batch_size=1,
                             shuffle=True,
                             drop_last=False)

    train(model, training_loader, validation_loader, validation_data, param_config)


if __name__ == '__main__':
    main()
