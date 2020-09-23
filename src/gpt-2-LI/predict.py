
from torch.utils.data import DataLoader
from data.load_data import get_wili_data

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

def predict(model, validation_loader, validation_data, config):
    model.eval()

    accuracy = validate_paragraphs(model, validation_data, validation_loader, subset=False)
    print("Validation Accuracy: {}".format(accuracy))

    #write_results((avg_train_loss, val_loss, val_accuracy), model_type+"_")
    print("Iterators Done")

def main():

    gpt_config = GPT2Config()
    param_config = config()

    model = GPT2LMHeadModel(gpt_config)
    # Load Data
    _, validation_data = get_wili_data(param_config)

    validation_loader = DataLoader(validation_data,
                             batch_size=1,
                             shuffle=False,
                             drop_last=False)

    if param_config.model_checkpoint is not None:
        with open(param_config.model_checkpoint, 'rb') as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)
            print("Model Loaded From: {}".format(param_config.model_checkpoint))

    model = model.to(device)
    predict(model, validation_loader, validation_data, param_config)


if __name__ == '__main__':
    main()
