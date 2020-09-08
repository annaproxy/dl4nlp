from tokenizers import ByteLevelBPETokenizer
from tokenizers.decoders import ByteLevel
NAME = 'unicode-medium'
LOAD = True 

if LOAD:
    # For loading
    tokenizer = ByteLevelBPETokenizer('{0}-vocab.json'.format(NAME), '{0}-merges.txt'.format(NAME))

else: 
    paths = ['../wili-2018/x_train.txt', '../wili-2018/x_test.txt']

    # For training
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=paths, vocab_size=20000, min_frequency=2)
    tokenizer.save('.',NAME)



# G is spatie: https://github.com/huggingface/tokenizers/issues/166
Dutch = tokenizer.encode("Mijn naam is Anna Langedijk en ik hou van groen, geel en blauw.")
Mand = tokenizer.encode("座羅馬天主教教堂,供奉聖伯多祿,位於同名的廣場上.該市鎮總面積11.61平方公里,2009年時的人口為323人.該市鎮總面積38.28平方公里,2009年時的.")
Hindi = tokenizer.encode("सफारी उद्यान स्थित है । यह 445 हैक्टेयर में फैला है । प्राकृतिक सुंदरता के बीच")
Russian = tokenizer.encode("Муамар Каддафи предложил одновременно вывести французские и ливийские войска из")

decoder = ByteLevel()


print(decoder.decode(Dutch.tokens))
print(decoder.decode(Mand.tokens))
print(decoder.decode(Hindi.tokens))
print(decoder.decode(Russian.tokens))


