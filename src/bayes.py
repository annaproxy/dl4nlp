import numpy as np
import csv 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from collections import defaultdict 


lang_dict = {}
fam_dict = {}
alphabet_dict = {}

with open('lstm/data/wili-2018/labels.csv', 'r') as f: 
    for line in f.readlines()[1:]:
        s = line.split(';')
        code = s[0]
        lang_dict[code] = s[1]
        fam_dict[code] = s[5]
        alphabet_dict[code] = s[6]
        

all_lans = np.array(list(sorted([k for k in lang_dict])))

def build_confusion(analysis):
    confused_bothways = defaultdict(float)
    for lan in all_lans:
        most_confused = sorted(analysis.stats_language(lan, False).items(), key=lambda x:x[1])
        for confused_lan, amt in most_confused:
            confused_bothways[(lan, confused_lan)] += amt
            confused_bothways[(confused_lan, lan)] += amt
    return confused_bothways

def f1_fam(analysis):
    f1s_per_fam = defaultdict(float)
    precision_per_fam = defaultdict(float)
    recall_per_fam = defaultdict(float )
    total_per_fam = defaultdict(float )
    for lan, f1 in analysis.f1s.items():
        f1s_per_fam[fam_dict[lan]] += f1 
        total_per_fam[fam_dict[lan]] += 1 
    for lan, f1 in analysis.precisions.items():
        precision_per_fam[fam_dict[lan]] += f1
    for lan, f1 in analysis.recalls.items():
        recall_per_fam[fam_dict[lan]] += f1 

    f1s_per_fam = sorted(f1s_per_fam.items(), key =lambda x:x[1]/ total_per_fam[x[0]])
    for lan, score in f1s_per_fam:
        print(lan, '(' + str(int(total_per_fam[lan])) + ')', round(score / total_per_fam[lan],3), round( precision_per_fam[lan] / total_per_fam[lan], 3) ,
            round(recall_per_fam[lan]/ total_per_fam[lan], 3) )


def f1(analysis, amt=10):
    sorted_f1s = sorted(analysis.f1s.items(), key=lambda x:x[1])
    sorted_f1s = [z for z in sorted_f1s if z[1] > 0]
    print('Languages with lowest f1')
    print('Lan, F1, Precision, Recall')
    for lan, f1 in sorted_f1s[:amt]:
        print(lang_dict[lan], '&', fam_dict[lan], '&',  round(f1,3), '&',  round(analysis.precisions[lan],3), '&', round(analysis.recalls[lan],3), lan)

    perfect = 0
    for _, f1 in sorted_f1s: 
        if f1 > 0.99: perfect += 1
    print("Perfect F1-scores", perfect)

class DeterministicAnalysis():
    def __init__(self, filename='Det_results.csv', readable_data_filename = 'lstm/data/wili-2018/x_test_clean.txt', 
            twitter=False, index_file=None):
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
                index = int(row[0])
                results_dict[index] = dict()
                results_dict[index]['prediction'] = row[1].strip()
                results_dict[index]['label'] = row[2].strip()
        self.results_dict = results_dict
        if twitter:
            self.id_to_array = dict()
            self.array_to_id = defaultdict(lambda:-1)
            with open(index_file, 'r') as f: 
                indices = eval(f.read().strip())
                for i, index in enumerate(indices):
                    self.id_to_array[index] = i 
                    self.array_to_id[i] = index
            lans = set() 
            for i in self.results_dict:
                p = self.results_dict[i]['label']
                lans.add(p)
            lans = list(sorted(lans))
            all_lans_twitter = list()
            for lan in all_lans: 
                if lan in lans: 
                    all_lans_twitter.append(lan) 


       
        self.all_lans = all_lans if not twitter else all_lans_twitter
        
        self.predictions = [results_dict[z]['prediction'] for z in results_dict]
        self.labels = [results_dict[z]['label'] for z in results_dict]
        self.f1s = { lan:f1 for lan, f1 in zip(self.all_lans, f1_score(self.labels, self.predictions, average=None, labels=self.all_lans)) }
        self.precisions = { lan:f1 for lan, f1 in zip(self.all_lans, precision_score(self.labels, self.predictions, average=None, labels=self.all_lans)) } 
        self.recalls = { lan:f1 for lan, f1 in zip(self.all_lans, recall_score(self.labels, self.predictions, average=None, labels=self.all_lans)) } 

        print('Micro', f1_score(self.labels, self.predictions, average='micro'))
        print('Macro',f1_score(self.labels, self.predictions, average='macro'))
        print('Accuracy', accuracy_score(self.labels, self.predictions))


    def stats_language(self, lan, true_label=True):
        confusion_amount = defaultdict(float)
        amt = 0
        for k in self.results_dict:
            true = self.results_dict[k]['label']
            pred = self.results_dict[k]['prediction']
            if true != pred:
                if (true_label and true ==lan) or (not true_label and pred ==lan):
                    amt += 1
                    confusion_amount[true if not true_label else pred] += 1
        print(amt)
        return confusion_amount

class BayesianAnalysis():
    def __init__(self, filename='Bayesian_Results_gpt.csv', readable_data_filename = 'lstm/data/wili-2018/x_test_clean.txt',
            twitter=False, index_file=None):
        self.twitter = twitter 

        results_dict = {}
        data_dict = {} 
        with open(readable_data_filename, 'r') as f: 
            for i, line in enumerate(f.readlines()):
                data_dict[i] = line
        with open(filename, 'r') as f: 
            reader = csv.reader(f, delimiter=';')
            next(reader)
            for row in reader: 

                results_dict[int(row[0])] = dict() 
                results_dict[int(row[0])]['prediction'] = row[1].strip()
                results_dict[int(row[0])]['label'] = row[2].strip()
                results_dict[int(row[0])]['means'] = eval(row[3])
                results_dict[int(row[0])]['sds'] = eval(row[4])
        self.data_dict = data_dict
        self.results_dict = results_dict 

        if twitter:
            self.id_to_array = dict()
            self.array_to_id = defaultdict(lambda:-1)
            with open(index_file, 'r') as f: 
                indices = eval(f.read().strip())
                for i, index in enumerate(indices):
                    self.id_to_array[index] = i 
                    self.array_to_id[i] = index
            lans = set() 
            for i in self.results_dict:
                p = self.results_dict[i]['label']
                lans.add(p)
            lans = list(sorted(lans))
            all_lans_twitter = list()
            for lan in all_lans: 
                if lan in lans: 
                    all_lans_twitter.append(lan) 
                
            


        self.all_lans = all_lans if not twitter else all_lans_twitter
        self.predictions = [results_dict[z]['prediction'] for z in results_dict]
        self.labels = [results_dict[z]['label'] for z in results_dict]
        self.languages = np.array(sorted(list(set(self.labels))))

        self.f1s = { lan:f1 for lan, f1 in zip(self.all_lans, f1_score(self.labels, self.predictions, average=None, labels=self.all_lans)) }
        self.precisions = { lan:f1 for lan, f1 in zip(self.all_lans, precision_score(self.labels, self.predictions, average=None, labels=self.all_lans)) } 
        self.recalls = { lan:f1 for lan, f1 in zip(self.all_lans, recall_score(self.labels, self.predictions, average=None, labels=self.all_lans)) } 

        print('Micro', f1_score(self.labels, self.predictions, average='micro'))
        print('Macro',f1_score(self.labels, self.predictions, average='macro'))
        print('Accuracy', accuracy_score(self.labels, self.predictions))

    def uncertainty(self, type='highest',amount=1, plot=False):
        all_uncertainties = np.array([ np.mean(self.results_dict[z]['sds'])  for z in self.results_dict  ])
        
        if type=='highest': 
            maximum_indices = all_uncertainties.argsort()[::-1][:amount]
        else: 
            maximum_indices = all_uncertainties.argsort()[:amount]
        
        if plot:
            for biggest_index in maximum_indices:
                biggest = all_uncertainties[biggest_index]
                if self.twitter:
                    biggest_index = self.array_to_id[biggest_index]
                
                print('Biggest avg sd', biggest)
                print('Label:', lang_dict[self.results_dict[biggest_index]['label']])
                print('Index:', biggest_index)

                print('Text:', self.data_dict[biggest_index])
                self.datapoint_stats(biggest_index)
                print('===')
        return maximum_indices


    def datapoint_stats(self, datapoint, cutoff=10, save_to_file=None):
        #print(lang_dict[self.results_dict[datapoint]['label']], lang_dict[self.results_dict[datapoint]['prediction']] )
        all_means = np.array(self.results_dict[datapoint]['means'])
        all_sds = np.array(self.results_dict[datapoint]['sds'])
        maximum_indices = all_means.argsort()[::-1][:cutoff]
        print(len(all_means), len(self.languages), maximum_indices)
        print('Mean of sds', np.mean(all_sds))
        xs = [lang_dict[z] for z in all_lans[maximum_indices]]
        ys = all_means[maximum_indices]
        fill = all_sds[maximum_indices]
        plt.figure()
        plt.plot(xs,ys)
        plt.ylim(-0.1,1)
        plt.fill_between(xs, ys-fill, ys+fill,alpha=0.2)
        #plt.errorbar(xs, ys yerr=,fill ecolor='green',linewidth=2.0, elinewidth=0.6)
        plt.xticks(rotation=80)
        plt.ylabel("Softmax probability")
        if save_to_file is None:
            plt.show()
        else: 
            plt.savefig(save_to_file, bbox_inches='tight')



    def language_stats(self, lan, cutoff=10, true_class=True, plot=False):
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
        #print(np.mean(all_sds))

        if plot:
            maximum_indices = all_means.argsort()[::-1][:cutoff]

            xs = [lang_dict[z] for z in self.languages[maximum_indices]]
            ys = all_means[maximum_indices]
            fill = all_sds[maximum_indices]
            plt.plot(xs,ys)
            plt.fill_between(xs, ys-fill, ys+fill,alpha=0.2)
            #plt.errorbar(xs, ys, yerr=fill, ecolor='green',linewidth=2.0, elinewidth=0.6)
            plt.xticks(rotation=80)
            plt.show()
        return np.mean(all_sds)

