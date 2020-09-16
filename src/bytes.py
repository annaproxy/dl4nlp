from tokenizers import ByteLevelBPETokenizer
from tokenizers.decoders import ByteLevel

NAME = 'unicode-medium'
LOAD = True 
train_path = 'data/wili-2018/x_train_sub.txt'
val_path = 'data/wili-2018/x_val_sub.txt'
test_path = 'data/wili-2018/x_test.txt'

if LOAD:
    # For loading
    tokenizer = ByteLevelBPETokenizer('{0}-vocab.json'.format(NAME), '{0}-merges.txt'.format(NAME))

else: 
    paths = [train_path, test_path]

    # For training
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=paths, vocab_size=20000, min_frequency=2)
    tokenizer.save('.',NAME)

# G is spatie: https://github.com/huggingface/tokenizers/issues/166
#Dutch = tokenizer.encode("Mijn naam is Anna Langedijk en ik hou van groen, geel en blauw.")
#Mand = tokenizer.encode("座羅馬天主教教堂,供奉聖伯多祿,位於同名的廣場上.該市鎮總面積11.61平方公里,2009年時的人口為323人.該市鎮總面積38.28平方公里,2009年時的.")
#Hindi = tokenizer.encode("सफारी उद्यान स्थित है । यह 445 हैक्टेयर में फैला है । प्राकृतिक सुंदरता के बीच")
#Russian = tokenizer.encode("Муамар Каддафи предложил одновременно вывести французские и ливийские войска из")

def write_bpe_file(in_path, out_path):
    with open(in_path, 'r') as f:
        lines = f.readlines()
        with open(out_path, 'w') as writer:
            for line in lines: 
                bp_ids = tokenizer.encode(line).ids
                writer.write(' '.join([str(z) for z in bp_ids]))
                writer.write('\n')

write_bpe_file(train_path,'data/wili-2018/x_bytes_train_sub.txt')
write_bpe_file(val_path,'data/wili-2018/x_bytes_val_sub.txt')

#decoder = ByteLevel()
#print(decoder.decode(Dutch.tokens))
#print(decoder.decode(Mand.tokens))
#print(decoder.decode(Hindi.tokens))
#print(Russian.tokens, Russian.ids)


