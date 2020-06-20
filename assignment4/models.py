# models.py

import torch
import torch.nn as nn
import numpy as np
import collections
from torch import optim
import random
import time
from torch.utils.data import DataLoader
#####################
# MODELS FOR PART 1 #
#####################

class RNN(nn.Module):
    def __init__(self, vocab_size, input_size=64, hidden = 128, num_class=2):
        super(RNN, self).__init__()
        
        self.embeddings = nn.Embedding(vocab_size, input_size)
        self.gru = nn.GRU(input_size, hidden, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden, num_class)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        '''
            x: size[batch, seq_len]
            x_embed: [batch, seq_len, input_size]
            output: classifer[]
        '''
        x_embed = self.embeddings(x) # [batch, seq_len, input_size]
        output, h_n = self.gru(x_embed) # h_n [num_layers * num_directions, batch, hidden_size]
        h_n = h_n.squeeze() # [batch, hidden_size]
        logits = self.linear(h_n)
        return logits


class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1


class RNNClassifier(ConsonantVowelClassifier):
    def __init__(self, word_indexer):
        super(RNNClassifier,self).__init__()
        self.indexer = word_indexer
        self.vocab_size = len(word_indexer)
        self.lossFunc = nn.CrossEntropyLoss()
        self.model = RNN(self.vocab_size)

    def predict(self, context):
        context_list = list(context)
        context_list = [self.indexer.index_of(c) for c in context_list]
        context_list_tensor = torch.tensor([context_list])
        probs = self.model(context_list_tensor)

        return torch.argmax(probs)

class Text8Dataset(torch.utils.data.Dataset):
    def __init__(self, cons_exs, vowel_exs, indexer):
        super(Text8Dataset, self).__init__()
        self.cons_exs = cons_exs
        self.vowel_exs = vowel_exs
        self.indexer = indexer
        self.preprocess()

    def preprocess(self):
        '''
            example(List), label(0/1)
        '''
        self.exs_characters = []
        self.exs_labels = []
        for cons_ex in self.cons_exs:
            cons_list = list(cons_ex)
            cons_list = [self.indexer.index_of(c) for c in cons_list]
            self.exs_characters.append(cons_list)
            self.exs_labels.append(0)
        for vowel_ex in self.vowel_exs:
            vowel_list = list(vowel_ex)
            vowel_list = [self.indexer.index_of(c) for c in vowel_list]
            self.exs_characters.append(vowel_list)
            self.exs_labels.append(1)
        self.exs_characters = torch.tensor(self.exs_characters)
        self.exs_labels = torch.tensor(self.exs_labels)


    def __getitem__(self, index):
        return self.exs_characters[index], self.exs_labels[index]
    
    def __len__(self):
        return len(self.exs_characters)
        
def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    BATCH_SIZE = 32
    EPOCHS = 10
    train_dset = Text8Dataset(train_cons_exs, train_vowel_exs, vocab_index)
    train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True)

    classifier = RNNClassifier(vocab_index)
    initial_lr = 1e-3
    optimizer = optim.Adam(classifier.model.parameters(), lr = initial_lr)

    t0=time.time()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        num_epoches = 0
        for batch_ex, batch_lb in train_loader:
            '''
            batch_ex: [batch, seq_lenth]
            '''
            num_epoches+=1

            classifier.model.train()
            optimizer.zero_grad()

            logits = classifier.model.forward(batch_ex)
            
            myloss = classifier.lossFunc(logits, batch_lb)
            total_loss+=myloss
            myloss.backward()
            optimizer.step()

        total_loss/=num_epoches
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    print("training time:", time.time()-t0)
    return classifier
    # raise Exception("Implement me")


#####################
# MODELS FOR PART 2 #
#####################

class RNNLM(nn.Module):
    def __init__(self, vocab_size, input_size=128, hidden = 128):
        super(RNNLM, self).__init__()
        
        self.embeddings = nn.Embedding(vocab_size, input_size)
        self.gru = nn.GRU(input_size, hidden, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden, vocab_size)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, h_0=None):
        '''
            x: size[batch, seq_len]
            x_embed: [batch, seq_len, input_size]
            output: classifer[batch,  vocab_size]
        '''
        x_embed = self.embeddings(x) # [batch, seq_len, input_size]
        if h_0 is not None:
            output, h_n = self.gru(x_embed, h_0)
        else:
            output, h_n = self.gru(x_embed) # h_n [num_layers * num_directions, batch, hidden_size]
                                        # output [batch, seq_len, num_directions * hidden_size]
        logits = self.linear(output) 

        return logits, h_n # logits [batch, seq_len, vocab_size]

class Text8LMDataset(torch.utils.data.Dataset):
    def __init__(self, train_text, indexer, chunk_size):
        super(Text8LMDataset, self).__init__()
        self.text = list(train_text)
        self.indexer = indexer
        self.chunk_size = chunk_size
        self.preprocess()

    def preprocess(self):
        '''
            example(List), label(0/1)
        '''
        self.exs_texts = []
        self.exs_labels = []
        self.chunk_text = [self.text[i:i+self.chunk_size] for i in range(0, len(self.text),self.chunk_size)]
        space_idx = self.indexer.index_of(' ')
        for chunk in self.chunk_text:
            while len(chunk)!=self.chunk_size:
                chunk.append(' ')
            chunk_index = [self.indexer.index_of(c) for c in chunk]
            self.exs_labels.append(chunk_index)
            self.exs_texts.append([space_idx]+chunk_index[:-1])
            
        self.exs_characters = torch.tensor(self.exs_texts)
        self.exs_labels = torch.tensor(self.exs_labels)

    def __getitem__(self, index):
        return self.exs_characters[index], self.exs_labels[index]
    
    def __len__(self):
        return len(self.exs_characters)


class LanguageModel(object):

    def get_log_prob_single(self, next_char, context):
        """
        Scores one character following the given context. That is, returns
        log P(next_char | context)
        The log should be base e
        :param next_char:
        :param context: a single character to score
        :return:
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context):
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return:
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_log_prob_single(self, next_char, context):
        return np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self, vocab_index):
        # raise Exception("Implement me")
        self.indexer = vocab_index
        self.vocab_size = len(vocab_index)
        self.lossFunc = nn.CrossEntropyLoss()
        self.model = RNNLM(self.vocab_size)
        self.neglogsoftmax= nn.CrossEntropyLoss()

    def get_log_prob_single(self, next_char, context):
        print(context)
        input()
        raise Exception("Implement me")

    def get_log_prob_sequence(self, next_chars, context):
        '''
        context: [seq_len, vocab_size]
        '''
        # print("char:",next_chars, "context:",context)
        next_index= [self.indexer.index_of(c) for c in next_chars]
        next_tensor = torch.tensor(next_index) # [seq_len]

        context_index= [self.indexer.index_of(c) for c in context]
        context_tensor = torch.tensor(context_index).unsqueeze(0) # [batch=1, seq_len]

        logits, h_n = self.model.forward(context_tensor) # logits [batch=1, seq_len, vocab_size]
        logits = logits[:,-1:,:]
        log_probs = 0
        for i, n in enumerate(next_index):
            n = torch.tensor(n).unsqueeze(0)
            logits = logits.squeeze(0)

            log_prob = - self.neglogsoftmax(logits, n)
            log_probs += log_prob.item()
            n = n.unsqueeze(0)
            logits, h_n = self.model.forward(n, h_n)
            

        return log_probs


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    
    BATCH_SIZE = 16
    CHUNK_SIZE = 50
    VOCAB_SIZE = len(vocab_index)
    EPOCHS = 15
    train_dset = Text8LMDataset(train_text, vocab_index, chunk_size=CHUNK_SIZE)
    train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True)

    LM = RNNLanguageModel(vocab_index)
    initial_lr = 1e-3
    optimizer = optim.Adam(LM.model.parameters(), lr = initial_lr)

    t0=time.time()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        num_epoches = 0
        for batch_input, batch_output in train_loader:
            '''
            batch_ex: [batch, seq_lenth]
            '''
            # print(batch_input[0], batch_output[0])
            
            num_epoches+=1

            LM.model.train()
            optimizer.zero_grad()

            logits,_ = LM.model.forward(batch_input) # logits [batch, seq_len, vocab_size]
            logits = logits.view(BATCH_SIZE*CHUNK_SIZE, VOCAB_SIZE)
            batch_output=batch_output.view(BATCH_SIZE*CHUNK_SIZE)
            # print(logits.shape)
            # print(batch_output.shape)
            # log_prob = get_log_prob_sequence()
            myloss = LM.lossFunc(logits, batch_output)
            total_loss+=myloss
            myloss.backward()
            optimizer.step()

        total_loss/=num_epoches
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    print("training time:", time.time()-t0)
    return LM
