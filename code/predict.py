import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.optim as optim
from LSTM import Model
from utils.utils import *

from data.load_data import get_wili_data
from utils.config import LSTM_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(model, validation_loader, validation_data, config):
    model.eval()

    accuracy = validate_paragraphs(model, validation_data, validation_loader, subset=False)
    print("Validation Accuracy: {}".format(accuracy))

    #write_results((avg_train_loss, val_loss, val_accuracy), model_type+"_")
    print("Iterators Done")

def main():

    config = LSTM_config()

    model = Model(config.input_dim, config.embedding_dim,
                  config.hidden_dim, config.num_layers, bidirectional=False)

    # Load Data
    _, validation_data = get_wili_data(config)


    validation_loader = DataLoader(validation_data,
                             batch_size=1,
                             shuffle=False,
                             drop_last=False)

    if config.model_checkpoint is not None:
        with open(config.model_checkpoint, 'rb') as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)
            print("Model Loaded From: {}".format(config.model_checkpoint))

    model = model.to(device)
    predict(model, validation_loader, validation_data, config)


if __name__ == '__main__':
    main()
