# models.py
import time
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
from tqdm import tqdm
import torch.nn.functional as F

# random.seed(0)
num_classes = 2

class NeuralNetwork(nn.Module):
    def __init__(self, word_embeddings=None, embed=50, hidden=128, output=2):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.
        :param input: size of input (integer)
        :param hidden: size of hidden layer(integer)
        :param output: size of output (integer), which should be the number of classes
        """
        super(NeuralNetwork, self).__init__()
        if word_embeddings is not None:
            vocab = len(word_embeddings.vectors)
            # self.embeddings = nn.Embedding(vocab, embed)
            self.embeddings =nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings.vectors), freeze=False)
        self.V = nn.Linear(embed, hidden)
        self.g = nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)
        self.V2 = nn.Linear(hidden, hidden)
        self.g2 = nn.Tanh()
        self.W = nn.Linear(hidden, output)
        # self.softmax = nn.Softmax()
        # self.log_softmax = nn.LogSoftmax()
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x):
        """
        :param x: a [input]-sized vector of input sentence
        :return: an [output]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        if self.embeddings is not None :
            sent_embedding = self.embeddings(x) # x is word idx list
            x_mean = torch.mean(sent_embedding, dim=1, keepdim=False).float()
            return self.W(self.g2(self.V2(self.dropout(self.g(self.V(x_mean))))))
        else:
            return self.W(self.g2(self.V2(self.g(self.V(x))))) # x is averaged sentence embedding


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, ex_words: List[str]):
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]):
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, word_embeddings: WordEmbeddings):
        SentimentClassifier.__init__(self)
        self.word_indexer = word_embeddings.word_indexer
        self.INPUT_SIZE = word_embeddings.get_embedding_length()
        self.HIDDEN_SIZE= 256
        self.OUTPUT_SIZE= num_classes
        # self.lossFunc = nn.NLLLoss()
        self.lossFunc = nn.CrossEntropyLoss()
        self.model = NeuralNetwork(word_embeddings, self.INPUT_SIZE, self.HIDDEN_SIZE, self.OUTPUT_SIZE)

    def predict(self, ex_words: List[str]):
        ex_words_idx = [max(1, self.word_indexer.index_of(word)) for word in ex_words]
        # sent_embedding = self.getSentenceEmbedding(ex_words_idx)
        ex_words_tensor=torch.tensor([ex_words_idx])
        y_probs = self.model.forward(ex_words_tensor)
        return torch.argmax(y_probs)

    def loss(self, probs, target):
        return self.lossFunc(probs, target)


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    NSC = NeuralSentimentClassifier(word_embeddings)

    sent2wordIdx = {}
    for i in range(len(train_exs)):
        wordList = train_exs[i].words
        lis = []
        for word in wordList:
            idx = NSC.word_indexer.index_of(word)
            lis.append(max(idx, 1))
        sent2wordIdx[i] = lis


    epochs = 16 # 16
    batch_size= 128

    initial_learning_rate = 0.01
    optimizer = optim.Adam(NSC.model.parameters(), lr=initial_learning_rate)

    ex_indices = [i for i in range(0,len(train_exs))]
    # random.shuffle(ex_indices)

    # batching
    
    for epoch in range(0, epochs):
        # use small dataset #
        # data_size = 6000
        # ex_indices = ex_indices[:data_size]
        random.shuffle(ex_indices)
        total_loss = 0.0
        batch_x = []
        batch_y = []
        pad_length=50
        t1=time.time()
        for idx in ex_indices:
            if len(batch_x)<batch_size:
                sent_pad = [0]*pad_length
                sent = sent2wordIdx[idx]
                # padding
                sent_pad[:min(pad_length,len(sent))]=sent[:min(pad_length,len(sent))]
                batch_x.append(sent_pad)
                y = train_exs[idx].label
                batch_y.append(y)

            else:   # len(batch_x) = batch_size
                NSC.model.train()
                optimizer.zero_grad()

                # print(batch_x)
                batch_x = torch.tensor(batch_x)
                probs = NSC.model.forward(batch_x)
                target = torch.tensor(batch_y)
                myloss = NSC.loss(probs, target)
                total_loss += myloss
                
                myloss.backward()
                optimizer.step()
                batch_x = []
                batch_y = []

        t2=time.time()
        # print("time for a epoch:", t2-t1)
        total_loss /= len(train_exs)
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    # no batching
    # for epoch in range(0, epochs):
    #     # use small dataset #
    #     # data_size = 1000
    #     # ex_indices = ex_indices[:data_size]
    #     #                   #
    #     total_loss = 0.0
    #     for idx in tqdm(ex_indices):
    #         sent = sent2wordIdx[idx]
    #         # x = NSC.getSentenceEmbedding(sent)
    #         y = train_exs[idx].label

    #         NSC.model.train()
    #         optimizer.zero_grad()
    #         x=torch.tensor([sent])
    #         probs = NSC.model.forward(x)
    #         # loss = torch.neg(probs).dot(y_onehot)
    #         target = torch.tensor([y])
    #         myloss = NSC.loss(probs, target)
    #         # print(probs, target, loss)
    #         total_loss += myloss
    #         # Computes the gradient and takes the optimizer step
    #         myloss.backward()
    #         optimizer.step()
    #     total_loss /= len(train_exs)
    #     print("Total loss on epoch %i: %f" % (epoch, total_loss))


    return NSC

