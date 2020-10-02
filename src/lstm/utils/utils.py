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


def validate_paragraphs(model, validation_data, validation_loader, textfile = 'data/twitter/x_twituser.txt' ,#_val_sub.txt',
    save_classification_report=True, subset=True):
    n_batches = len(validation_loader)
    if subset: n_batches = 1000

    with open(textfile) as f:
        pars = f.readlines()

    validation_data.predict_paragraph(True)
    model.eval()
    model = model #.cpu()
    y_pred = []; y_true = [];
    accuracies = []
    #print(len(validation_loader))
    wrong_english_indices = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validation_loader):
            #print(inputs.shape, labels.shape)
            if inputs.shape[1] < 1: continue

            inputs = inputs.to(device).squeeze(0).long() #.cpu()
            labels = labels.to(device).squeeze(0) #.cpu()
            #if not validation_data.lang_to_idx["eng"] == labels[0].item():
        #        continue
            #for n in range(30):
            logits = model(inputs, False)
            probs = get_mean_softmax(logits)
            #output = torch.argmax(logits,dim=1).view(-1)
            prediction = torch.argmax(probs)
            #prediction = get_paragraph_prediction(output, labels)
            #print(prediction)
            #    print(validation_data.idx_to_lang[prediction.item()])
            #raise ValueError()
            y_pred.append(prediction.item())
            y_true.append(labels[0].item())
            

            #lang_pred = prediction.item()
            #lang_true = labels[0].item()
            #if lang_pred == validation_data.lang_to_idx['eng']:
            #    if lang_pred != lang_true:
            #        #print(i, validation_data.idx_to_lang[lang_true], pars[i])
            #        wrong_english_indices.append(i)
            correct = (prediction == labels[0].item()).float()
            if (labels[0].item() == 50 or prediction.item() == 50) and correct.item() < 1:
                print(correct.item(),i, validation_data.idx_to_lang[labels[0].item()],
                     validation_data.idx_to_lang[prediction.item()],
                     pars[i])
            """
            entropy, probs = get_entropy(logits)
            probs = probs.cpu().numpy()

            if correct:
                print("Correct classification Entropy: {}".format(entropy))
            if not correct:
                print("entropy of incorrect classification: {}".format(entropy))
                print("Language {} mistaken for {}.".format(validation_data.idx_to_lang[labels[0].item()], validation_data.idx_to_lang[prediction.item()]))
                #fig = plt.figure()
                #ax = fig.add_axes([0,0,1,1])
                langs = np.array([validation_data.idx_to_lang[i] for i in range(235)])
                #sns.set_theme(style="whitegrid")
                #ax = sns.barplot( data=[probs.cpu().numpy())
                #plt.show()
                #print(langs)
                pd_map = {l:[probs[i]] for i, l in enumerate(langs)}
                pd.DataFrame(probs, index=langs).plot.bar()
                plt.show()

                #ax.bar(langs,probs.cpu().numpy()[:10])
                #plt.show()
            """
            accuracies.append(correct.item())
            if i == n_batches: break
    #print(len(wrong_english_indices))

    #with open("indices_fucked_test.txt", "w") as f:
    #    np.savetxt(f, np.array(wrong_english_indices))

    #with open ('confmatrix2.txt', 'w') as f:
    #    np.savetxt(f, confusion_matrix(y_true, y_pred).astype(int), fmt='%i', delimiter=',')
    accuracy = round(np.sum(accuracies)/(n_batches),4)

    print(accuracy)
    def rowIndex(row):return row.name
    if save_classification_report:
        #target_names = validation_data.idx_to_lang
        df = pd.DataFrame(classification_report(y_true, y_pred,  output_dict=True)).transpose()
        df['lan_index'] = df.apply(rowIndex, axis=1) # for z in df.index]
        df['lan'] = df['lan_index'].map(lambda z:validation_data.idx_to_lang[int(z)] if len(z) < 4 else '')
        df.to_csv('classification_report_DROPOUT_BOY_CLEANED.csv', index= True)
        #print(df)
    return accuracy


def validate_uncertainty(model, validation_data, validation_loader):

    validation_data.predict_paragraph(True)
    #model.eval()
    model = model #.cpu()
    y_pred = []; y_true = [];
    accuracies = []

    wrong_english_indices = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to(device).squeeze(0) #.cpu()
            labels = labels.to(device).squeeze(0) #.cpu()
            #if not validation_data.lang_to_idx["eng"] == labels[0].item():
        #        continue

            datapoint_probs = torch.zeros(inputs.shape[0],235)
            for n in range(50):
                logits = model(inputs, True)
                probabilities = get_mean_softmax(logits)
                print(probabilities.shape)
                datapoint_probs[n,:] = probabilities
                raise ValueError()

            #predictions = torch.argmax(datapoint_probs, dim=1)
            standard_deviations = torch.std(datapoint_probs, dim=0)
            means = torch.mean(datapoint_probs, dim=0)
            sorted_predictions = torch.argsort(means)
            top_10 = sorted_predictions[-10:]


            output = torch.argmax(logits,dim=1).view(-1)
            prediction = get_paragraph_prediction(output, labels)
            #    print(validation_data.idx_to_lang[prediction.item()])
            #raise ValueError()
            y_pred.append(prediction.item())
            y_true.append(labels[0].item())

            #lang_pred = prediction.item()
            #lang_true = labels[0].item()
            #if lang_pred == validation_data.lang_to_idx['eng']:
            #    if lang_pred != lang_true:
            #        #print(i, validation_data.idx_to_lang[lang_true], pars[i])
            #        wrong_english_indices.append(i)
            correct = (prediction == labels[0].item()).float()
            entropy, probs = get_entropy(logits)
            probs = probs.cpu().numpy()

            if correct:
                print("Correct classification Entropy: {}".format(entropy))
            if not correct:
                print("entropy of incorrect classification: {}".format(entropy))
                print("Language {} mistaken for {}.".format(validation_data.idx_to_lang[labels[0].item()], validation_data.idx_to_lang[prediction.item()]))
                #fig = plt.figure()
                #ax = fig.add_axes([0,0,1,1])
                langs = np.array([validation_data.idx_to_lang[i] for i in range(235)])
                #sns.set_theme(style="whitegrid")
                #ax = sns.barplot( data=[probs.cpu().numpy())
                #plt.show()
                #print(langs)
                pd_map = {l:[probs[i]] for i, l in enumerate(langs)}
                pd.DataFrame(probs, index=langs).plot.bar()
                plt.show()

                #ax.bar(langs,probs.cpu().numpy()[:10])
                #plt.show()



            accuracies.append(correct.item())
    #print(len(wrong_english_indices))

    #with open("indices_fucked_test.txt", "w") as f:
    #    np.savetxt(f, np.array(wrong_english_indices))

    #with open ('confmatrix2.txt', 'w') as f:
    #    np.savetxt(f, confusion_matrix(y_true, y_pred).astype(int), fmt='%i', delimiter=',')
    accuracy = round(np.sum(accuracies)/(n_batches),4)

    print(accuracy)
    return accuracy
