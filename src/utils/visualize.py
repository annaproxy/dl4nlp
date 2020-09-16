import pandas as pd 
df = pd.read_csv('../classification_report.txt', delimiter='\t')
#df = pd.read_csv('../bryan_newest.txt', delimiter='\t')

lang_dict = {}
fam_dict = {}
alphabet_dict = {}


with open('../data/wili-2018/labels.csv', 'r') as f: 
    for line in f.readlines():
        s = line.split(';')
        code = s[0]
        lang_dict[code] = s[1]
        fam_dict[code] = s[5]
        alphabet_dict[code] = s[6]

df['language'] = df['lang'].map(lang_dict)
df['fam'] = df['lang'].map(fam_dict)
df['alphabet'] = df['lang'].map(alphabet_dict)

with pd.option_context('display.max_rows', None): 
    print(df.sort_values('f1-score')[['language', 'precision','recall','f1-score', 'fam','alphabet' ]])

