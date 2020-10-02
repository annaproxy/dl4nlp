import numpy as np
import torch
import datetime
import torch.nn as nn
from sklearn.metrics import classification_report
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def get_accuracy(outputs, labels):
    return torch.sum(outputs==labels).float() #/ labels.shape[0]

def get_paragraph_prediction(outputs, labels):
    unique_preds = torch.unique(outputs)
    #print("unique_preds: ", unique_preds)
    counts = torch.tensor([(outputs==lang).float().sum() for lang in unique_preds])
    #print("counts: ", counts)
    max_lang = unique_preds[torch.argmax(counts)]
    return max_lang

def get_mean_softmax(outputs):
    probabilities = torch.softmax(outputs, dim=1)
    probabilities = torch.mean(probabilities, dim=0)
    return probabilities

def validate_paragraphs(model, validation_data, validation_loader, save_classification_report=True, subset=True):
    n_batches = len(validation_loader)
    if subset: n_batches = 500
    validation_data.predict_paragraph(True)
    y_pred = []; y_true = [];
    accuracies = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to(device).squeeze(0)
            #print(inputs.shape)
            labels = labels.to(device).squeeze(0)
            logits = model(inputs, eval=True)
            #output = torch.argmax(output,dim=1).view(-1)
            #prediction = get_paragraph_prediction(output, labels)
            probs = get_mean_softmax(logits)
            #output = torch.argmax(logits,dim=1).view(-1)
            prediction = torch.argmax(probs)
            y_pred.append(prediction.item()); y_true.append(labels[0].item());
            correct = (prediction == labels[0].item()).float()
            #print("prediction: ", prediction, "label: ", labels[0].item())
            #print(correct)
            accuracies.append(correct.item())
            #print(i)
            if i == n_batches: break

    accuracy = round(np.sum(accuracies)/(n_batches),4)
    if save_classification_report:
        target_names = validation_data.languages
        df = pd.DataFrame(classification_report(y_true, y_pred, target_names=target_names, output_dict=True)).transpose()
        df.to_csv('classification_report_charclean.csv', index= True)
    return accuracy

def validate_paragraph(model, validation_data, validation_loader, textfile = 'data/wili-2018/x_val_sub.txt',
    save_classification_report=True, subset=True):
    n_batches = len(validation_loader)
    if subset: n_batches = 1000

    validation_data.predict_paragraph(True)
    model.eval()
    y_pred = []; y_true = [];
    accuracies = []
    wrong_english_indices = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to(device).squeeze(0) #.cpu()
            labels = labels.to(device).squeeze(0) #.cpu()

            logits = model(inputs, False)
            #log_alpha = model._bayesian._log_alpha
            #print(log_alpha)
            #raise ValueError()
            probs = get_mean_softmax(logits)
            #output = torch.argmax(logits,dim=1).view(-1)
            prediction = torch.argmax(probs)
            #prediction = get_paragraph_prediction(output, labels)

            y_pred.append(prediction.item())
            y_true.append(labels[0].item())

            #lang_pred = prediction.item()
            #lang_true = labels[0].item()
            #if lang_pred == validation_data.lang_to_idx['eng']:
            #    if lang_pred != lang_true:
            #        #print(i, validation_data.idx_to_lang[lang_true], pars[i])
            #        wrong_english_indices.append(i)
            correct = (prediction == labels[0].item()).float()
            accuracies.append(correct.item())
            if i == n_batches: break
    #print(len(wrong_english_indices))

    #with open("indices_fucked_test.txt", "w") as f:
    #    np.savetxt(f, np.array(wrong_english_indices))

    #with open ('confmatrix2.txt', 'w') as f:
    #    np.savetxt(f, confusion_matrix(y_true, y_pred).astype(int), fmt='%i', delimiter=',')
    accuracy = round(np.sum(accuracies)/(n_batches),4)

    print(accuracy)

    if save_classification_report:
        target_names = validation_data.languages
        df = pd.DataFrame(classification_report(y_true, y_pred, target_names=target_names, output_dict=True)).transpose()
        df.to_csv('classification_report_kl_lstm.csv', index= True)
    return accuracy
