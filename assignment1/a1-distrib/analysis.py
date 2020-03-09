f = open('perceptron_feats.txt', 'r')
word2weight = {}
for line in f.readlines():
    word, weight = line.split()[0], line.split()[1]
    word2weight[word] = weight
word2weight = sorted(word2weight.items(), key=lambda x:x[1])
print(word2weight[4000:8000])