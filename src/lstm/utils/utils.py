import numpy as np
import torch
import pandas as pd
import datetime
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
    #print("counts: ", counts)
    max_lang = unique_preds[torch.argmax(counts)]
    return max_lang

def get_mean_softmax(outputs):
    probabilities = torch.softmax(outputs, dim=1)
    probabilities = torch.mean(probabilities, dim=0)
    return probabilities

def get_entropy(outputs):
    probabilities = get_mean_softmax(outputs)
    probabilities = torch.tensor([1/235 for i in range(235)]).cuda()
    log_probs = torch.log(probabilities)
    entropy = -torch.sum(probabilities * log_probs)
    print(entropy)
    raise ValueError()
    return entropy, probabilities


def validate_paragraphs(model, validation_data, validation_loader, textfile = 'data/wili-2018/x_val_sub.txt',
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


def validate_uncertainty(model, validation_data, validation_loader):
    n_batches = len(validation_loader)
    validation_data.predict_paragraph(True)
    #model.eval()
    model = model #.cpu()
    y_pred = []; y_true = [];
    accuracies = []

    wrong_english_indices = []
    with open("Bayesian_Results_LSTM.csv", "w") as file:
        file.write("Data_index, prediction, label, means, std\n")

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to(device).squeeze(0) #.cpu()
            labels = labels.to(device).squeeze(0) #.cpu()
            #if not validation_data.lang_to_idx["eng"] == labels[0].item():
        #        continue

            datapoint_probs = torch.zeros(50,235)
            for n in range(50):
                logits = model(inputs, False)
                probabilities = get_mean_softmax(logits)
                datapoint_probs[n,:] = probabilities

            standard_deviations = torch.std(datapoint_probs, dim=0)
            means = torch.mean(datapoint_probs, dim=0)
            prediction = torch.argmax(means).item()
            sorted_predictions = torch.argsort(means)
            top_10 = sorted_predictions[-10:]
            print("means: ",means[top_10])
            print("std: ",standard_deviations[top_10])
            print()
            #raise ValueError()
            label = labels[0].item()

            #output = torch.argmax(logits,dim=1).view(-1)
            #    print(validation_data.idx_to_lang[prediction.item()])
            #raise ValueError()
            y_pred.append(prediction)
            y_true.append(label)

            #lang_pred = prediction.item()
            #lang_true = labels[0].item()
            #if lang_pred == validation_data.lang_to_idx['eng']:
            #    if lang_pred != lang_true:
            #        #print(i, validation_data.idx_to_lang[lang_true], pars[i])
            #        wrong_english_indices.append(i)
            correct = (prediction == labels)#.float()

            means = [round(mean, 6) for mean in means.cpu().numpy()]
            std = [round(std_i, 6) for std_i in standard_deviations.cpu().numpy()]
            with open("Bayesian_Results_LSTM.csv", "a") as file:
                file.write(str(i)+", "+validation_data.idx_to_lang[prediction]+", " + \
                            validation_data.idx_to_lang[label]+", "+str(means)+", " + \
                            str(std)+"\n")



            accuracies.append(correct)
    #print(len(wrong_english_indices))

    #with open("indices_fucked_test.txt", "w") as f:
    #    np.savetxt(f, np.array(wrong_english_indices))

    #with open ('confmatrix2.txt', 'w') as f:
    #    np.savetxt(f, confusion_matrix(y_true, y_pred).astype(int), fmt='%i', delimiter=',')
    accuracy = round(np.sum(accuracies)/(n_batches),4)

    print(accuracy)
    return accuracy
