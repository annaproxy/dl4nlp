import numpy as np
import torch
import pandas as pd
import datetime
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def write_results(results, result_name):
    train_loss, val_loss, val_accuracy = results
    with open("./logs/"+result_name+str(datetime.datetime.now())[4:16]+".csv", "w") as f:
        for i in range(len(train_loss)):
            f.write(str(train_loss[i])+","+ str(val_loss[i])+"," + str(val_accuracy[i])+"\n")

def get_accuracy(outputs, labels):
    return torch.sum(outputs==labels).float() 

def get_paragraph_prediction(outputs, labels):
    unique_preds = torch.unique(outputs)
    counts = torch.tensor([(outputs==lang).float().sum() for lang in unique_preds])
    max_lang = unique_preds[torch.argmax(counts)]
    return max_lang

def validate_model_old(model, validation_data, validation_loader, batch_size, save_classification_report=True):
    n_batches = len(validation_loader)
    model.eval()
    accuracies = []
    y_pred= []; y_true = [];
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to(device);
            labels = labels.to(device)
            output = model(inputs)
            output = torch.argmax(output,dim=1).view(-1)
            y_pred += output.tolist()
            y_true += labels.view(-1).tolist()
            accuracy = get_accuracy(output, labels)
            accuracies.append(accuracy.item())
    accuracy = round(np.sum(accuracies)/(n_batches*batch_size),3)
    if save_classification_report:
        with open("classification_report.txt", 'w') as file:
            target_names = validation_data.languages
            file.write(classification_report(y_true, y_pred, target_names=target_names))
    return accuracy

def validate_paragraphs(model, validation_data, validation_loader, textfile = 'data/wili-2018/x_val_sub.txt', 
    save_classification_report=True, subset=True):
    n_batches = len(validation_loader)
    if subset: n_batches = 1000
    
    #with open(textfile) as f:
    #    pars = f.readlines()

    validation_data.predict_paragraph(True)
    model.eval()
    model = model #.cpu()
    y_pred = []; y_true = [];
    accuracies = []

    wrong_english_indices = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to(device).squeeze(0) #.cpu()
            labels = labels.to(device).squeeze(0) #.cpu()
            output = model(inputs)
            output = torch.argmax(output,dim=1).view(-1)
            prediction = get_paragraph_prediction(output, labels)
            y_pred.append(prediction.item())
            y_true.append(labels[0].item())

            lang_pred = prediction.item()
            lang_true = labels[0].item()
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

    #print(accuracy)

    if save_classification_report:
        target_names = validation_data.languages
        df = pd.DataFrame(classification_report(y_true, y_pred, target_names=target_names, output_dict=True)).transpose()
        df.to_csv('classification_report_charclean.csv', index= True)
    return accuracy
