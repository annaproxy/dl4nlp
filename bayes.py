import numpy as np
import csv 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


lang_dict = {}
fam_dict = {}
alphabet_dict = {}

with open('src/lstm/data/wili-2018/labels.csv', 'r') as f: 
    for line in f.readlines():
        s = line.split(';')
        code = s[0]
        lang_dict[code] = s[1]
        fam_dict[code] = s[5]
        alphabet_dict[code] = s[6]

class Analysis():
    def __init__(self, filename='Bayesian_Results_gpt.csv', readable_data_filename = 'src/lstm/data/wili-2018/x_val_sub_clean.txt'):
        results_dict = {}
        data_dict = {} 
        with open(readable_data_filename, 'r') as f: 
            for i, line in enumerate(f.readlines()):
                data_dict[i] = line

        self.data_dict = data_dict

        with open(filename, 'r') as f: 
            reader = csv.reader(f, delimiter=';')
            next(reader)
            for row in reader: 

                results_dict[int(row[0])] = dict() 
                results_dict[int(row[0])]['prediction'] = row[1].strip()
                results_dict[int(row[0])]['label'] = row[2].strip()
                results_dict[int(row[0])]['means'] = eval(row[3])
                results_dict[int(row[0])]['sds'] = eval(row[4])

        self.results_dict = results_dict 
        self.predictions = [results_dict[z]['prediction'] for z in results_dict]
        self.labels = [results_dict[z]['label'] for z in results_dict]
        self.languages = np.array(sorted(list(set(self.labels))))
        print(f1_score(self.labels, self.predictions, average='micro'))
        print(f1_score(self.labels, self.predictions, average='macro'))
        print(accuracy_score(self.labels, self.predictions))


    def uncertainty(self, type='highest', amount=1):

        all_uncertainties = np.array([ np.mean(self.results_dict[z]['sds'])  for z in self.results_dict  ])
        if type=='highest': 
            maximum_indices = all_uncertainties.argsort()[::-1][:amount]
        else: 
            maximum_indices = all_uncertainties.argsort()[:amount]

        for biggest_index in maximum_indices:
            biggest = all_uncertainties[biggest_index]
            print('Biggest avg sd', biggest)
            print('Label:', lang_dict[self.results_dict[biggest_index]['label']])
            print('Text:', self.data_dict[biggest_index])
            self.datapoint_stats(biggest_index)
            print('===')


    def datapoint_stats(self, datapoint, cutoff=10):

        all_means = np.array(self.results_dict[datapoint]['means'])
        all_sds = np.array(self.results_dict[datapoint]['sds'])
        maximum_indices = all_means.argsort()[::-1][:cutoff]

        plt.errorbar([lang_dict[z] for z in self.languages[maximum_indices]], 
                    all_means[maximum_indices], yerr=all_sds[maximum_indices],
                    ecolor='green',linewidth=2.0, elinewidth=0.6)
        plt.xticks(rotation=80)
        plt.show()


    def language_stats(self, lan, cutoff=10, true_class=True):
        all_means = np.zeros(len(self.languages))
        all_sds = np.zeros(len(self.languages))
        amt = 0.0
        for z in self.results_dict:
            if (true_class and self.results_dict[z]['label'] == lan) or (not true_class and self.results_dict[z]['prediction'] == lan):
                all_means += self.results_dict[z]['means']
                all_sds += self.results_dict[z]['sds']
                amt += 1

        all_means /= amt 
        all_sds /= amt
        maximum_indices = all_means.argsort()[::-1][:cutoff]
        plt.errorbar([lang_dict[z] for z in self.languages[maximum_indices]], 
                    all_means[maximum_indices], yerr=all_sds[maximum_indices],
                    ecolor='green',linewidth=2.0, elinewidth=0.6)
        plt.xticks(rotation=80)
        plt.show()


#a = Analysis()
#a.language_stats('spa')