import csv 
lang_dict_readable = {}

with open('data/wili-2018/labels.csv', 'r') as f: 
    for line in f.readlines():
        s = line.split(';')
        code = s[0]
        lang_dict_readable[code] = s[1]

langdict = {0: 'ace', 1: 'afr', 2: 'als', 3: 'amh', 4: 'ang', 5: 'ara', 6: 'arg', 7: 'arz', 8: 'asm', 9: 'ast', 10: 'ava', 11: 'aym', 12: 'azb', 13: 'aze', 14: 'bak', 15: 'bar', 16: 'bcl', 17: 'be-tarask', 18: 'bel', 19: 'ben', 20: 'bho', 21: 'bjn', 22: 'bod', 23: 'bos', 24: 'bpy', 25: 'bre', 26: 'bul', 27: 'bxr', 28: 'cat', 29: 'cbk', 30: 'cdo', 31: 'ceb', 32: 'ces', 33: 'che', 34: 'chr', 35: 'chv', 36: 'ckb', 37: 'cor', 38: 'cos', 39: 'crh', 40: 'csb', 41: 'cym', 42: 'dan', 43: 'deu', 44: 'diq', 45: 'div', 46: 'dsb', 47: 'dty', 48: 'egl', 49: 'ell', 50: 'eng', 51: 'epo', 52: 'est', 53: 'eus', 54: 'ext', 55: 'fao', 56: 'fas', 57: 'fin', 58: 'fra', 59: 'frp', 60: 'fry', 61: 'fur', 62: 'gag', 63: 'gla', 64: 'gle', 65: 'glg', 66: 'glk', 67: 'glv', 68: 'grn', 69: 'guj', 70: 'hak', 71: 'hat', 72: 'hau', 73: 'hbs', 74: 'heb', 75: 'hif', 76: 'hin', 77: 'hrv', 78: 'hsb', 79: 'hun', 80: 'hye', 81: 'ibo', 82: 'ido', 83: 'ile', 84: 'ilo', 85: 'ina', 86: 'ind', 87: 'isl', 88: 'ita', 89: 'jam', 90: 'jav', 91: 'jbo', 92: 'jpn', 93: 'kaa', 94: 'kab', 95: 'kan', 96: 'kat', 97: 'kaz', 98: 'kbd', 99: 'khm', 100: 'kin', 101: 'kir', 102: 'koi', 103: 'kok', 104: 'kom', 105: 'kor', 106: 'krc', 107: 'ksh', 108: 'kur', 109: 'lad', 110: 'lao', 111: 'lat', 112: 'lav', 113: 'lez', 114: 'lij', 115: 'lim', 116: 'lin', 117: 'lit', 118: 'lmo', 119: 'lrc', 120: 'ltg', 121: 'ltz', 122: 'lug', 123: 'lzh', 124: 'mai', 125: 'mal', 126: 'map-bms', 127: 'mar', 128: 'mdf', 129: 'mhr', 130: 'min', 131: 'mkd', 132: 'mlg', 133: 'mlt', 134: 'mon', 135: 'mri', 136: 'mrj', 137: 'msa', 138: 'mwl', 139: 'mya', 140: 'myv', 141: 'mzn', 142: 'nan', 143: 'nap', 144: 'nav', 145: 'nci', 146: 'nds', 147: 'nds-nl', 148: 'nep', 149: 'new', 150: 'nld', 151: 'nno', 152: 'nob', 153: 'nrm', 154: 'nso', 155: 'oci', 156: 'olo', 157: 'ori', 158: 'orm', 159: 'oss', 160: 'pag', 161: 'pam', 162: 'pan', 163: 'pap', 164: 'pcd', 165: 'pdc', 166: 'pfl', 167: 'pnb', 168: 'pol', 169: 'por', 170: 'pus', 171: 'que', 172: 'roa-tara', 173: 'roh', 174: 'ron', 175: 'rue', 176: 'rup', 177: 'rus', 178: 'sah', 179: 'san', 180: 'scn', 181: 'sco', 182: 'sgs', 183: 'sin', 184: 'slk', 185: 'slv', 186: 'sme', 187: 'sna', 188: 'snd', 189: 'som', 190: 'spa', 191: 'sqi', 192: 'srd', 193: 'srn', 194: 'srp', 195: 'stq', 196: 'sun', 197: 'swa', 198: 'swe', 199: 'szl', 200: 'tam', 201: 'tat', 202: 'tcy', 203: 'tel', 204: 'tet', 205: 'tgk', 206: 'tgl', 207: 'tha', 208: 'ton', 209: 'tsn', 210: 'tuk', 211: 'tur', 212: 'tyv', 213: 'udm', 214: 'uig', 215: 'ukr', 216: 'urd', 217: 'uzb', 218: 'vec', 219: 'vep', 220: 'vie', 221: 'vls', 222: 'vol', 223: 'vro', 224: 'war', 225: 'wln', 226: 'wol', 227: 'wuu', 228: 'xho', 229: 'xmf', 230: 'yid', 231: 'yor', 232: 'zea', 233: 'zh-yue', 234: 'zho'}
langdict = {k:lang_dict_readable[v] for k,v in langdict.items()}
from collections import defaultdict 
conf_dict = defaultdict(int)
with open('confmatrix.txt', 'r') as f :
    yes = csv.reader(f,delimiter=',')
    for i, row in enumerate(yes):
        for j, no in enumerate(row):
            conf_dict[(langdict[i],langdict[j])] = int(no)

sorted_confdict = sorted(conf_dict.items(), key = lambda x:x[1], reverse=False)

for (l1,l2), v in sorted_confdict:
    if l1 != l2: 
        print(l1, 'mistook for', l2,':', v)
