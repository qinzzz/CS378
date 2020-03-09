# models.py

from sentiment_data import *
from utils import *
import numpy as np
from collections import Counter
from tqdm import tqdm
from nltk.corpus import stopwords
import random
import matplotlib.pyplot as plt

random.seed(0)

def sigmoid(x):
    # if x < 0:
    #     output= np.exp(x)/(1. + np.exp(x))
    # else:
    #     output= 1./(1. + np.exp( -x ))
    output= 1./(1. + np.exp( -x ))
    output =  max(output, 1e-8)
    output = min(output, 1-1e-8)
    return output

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, ex_words: List[str], add_to_indexer: bool=False) -> List[int]:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param ex_words: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return:
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        FeatureExtractor.__init__(self)
        self.indexer = indexer
        self.countAll=Counter()
        # raise Exception("Must be implemented")

    def get_indexer(self):
        return self.indexer

    def add_features(self, ex_words: List[str]):
        # stop = stopwords.words('english')
        stop=[]
        for word in ex_words:
            word.lower()
            if word not in stop:
                self.indexer.add_and_get_index(word)
                
    def extract_features(self, ex_words: List[str], add_to_indexer: bool=False) -> List[int]:
        if add_to_indexer:
            self.add_features(ex_words)

        # countSet = set(self.countAll)
        c = Counter()
        for word in ex_words:
            word.lower()
            if self.indexer.contains(word):
                key = self.indexer.index_of(word)
                c.update([key])

        # feature = [0] * self.indexer.__len__()
        # for word in ex_words:
        #     word.lower()
        #     if self.indexer.contains(word):
        #         feature[self.indexer.index_of(word)]+=1
        
        return list(c.items())


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        FeatureExtractor.__init__(self)
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def add_features(self, ex_words: List[str]):
        for i in range(len(ex_words)-1):
            wordPair = ex_words[i]+ex_words[i+1]
            self.indexer.add_and_get_index(wordPair)
        
        # unigram
        # for word in ex_words:
        #     word.lower()
        #     self.indexer.add_and_get_index(word)

    def extract_features(self, ex_words: List[str], add_to_indexer: bool=False) -> List[int]:
        if add_to_indexer:
            self.add_features(ex_words)

        c = Counter()
        for i in range(len(ex_words)-1):
            wordPair = ex_words[i]+ex_words[i+1]
            if self.indexer.contains(wordPair):
                key = self.indexer.index_of(wordPair)
                c.update([key])
        # for word in ex_words:
        #     word.lower()
        #     if self.indexer.contains(word):
        #         key = self.indexer.index_of(word)
        #         c.update([key])

        return list(c.items())


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!

    Now: TF-IDF
    """
    def __init__(self, indexer: Indexer):
        FeatureExtractor.__init__(self)
        self.indexer = indexer
        self.numOfDoc =0
        self.word2docCount = {}

    def get_indexer(self):
        return self.indexer

    def idf_extractor(self, train_exs):
        self.numOfDoc = len(train_exs)
        for ex in train_exs:
            for word in ex.words:
                key = self.indexer.index_of(word)
                if key in self.word2docCount:
                    self.word2docCount[key]+=1
                else:
                    self.word2docCount[key]=1

    def add_features(self, ex_words: List[str]):
        # stop = stopwords.words('english')
        stop=[]
        for word in ex_words:
            # word.lower()
            if word not in stop:
                self.indexer.add_and_get_index(word)

    def extract_features(self, ex_words: List[str], add_to_indexer: bool=False) -> List[int]:
        if add_to_indexer:
            self.add_features(ex_words)
        # term frequency
        c = Counter()
        for word in ex_words:
            # word.lower()
            if self.indexer.contains(word):
                key = self.indexer.index_of(word)
                c.update([key])

        feats = []
        total_freq = len(ex_words)
        for key, count in list(c.items()):
            tf = np.log(count/total_freq + 1)
            # tf = count/total_freq
            idf = -np.log((self.word2docCount[key]+1)/self.numOfDoc)
            feats.append((key, tf*idf))
            # feats.append((key, tf))

        return feats


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex_words: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, ex_words: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feat_extractor):
        SentimentClassifier.__init__(self)
        self.feat_extractor = feat_extractor
        self.indexer = self.feat_extractor.get_indexer()
        self.vocab_size = self.indexer.__len__()
        self.weights = np.zeros((self.vocab_size,))
        self.feature_dict = {}

    def get_feature(self, ex_words: List[str]) -> List[int]:
        ex_sent = ''.join(ex_words)
        if ex_sent not in self.feature_dict:
            f_x = self.feat_extractor.extract_features(ex_words)
            self.feature_dict[ex_sent] = f_x
        else:
            f_x = self.feature_dict[ex_sent]
        return f_x

    def predict(self, ex_words: List[str]) -> int:
        '''
         y_pred = sign(w.T * f(x)) 
        
        '''
        fx= self.get_feature(ex_words)
        # y_pred = self.weights.dot(f_x)
        wfx = 0
        for key,val in fx:
            wfx += self.weights[key] * val
        y_ret = 1 if wfx >=0.5 else 0
        return y_ret
    
    def update(self, ex_words, y, y_pred, alpha):
        fx= self.get_feature(ex_words)
        # y_pred = self.weights.dot(f_x)

        for key,val in fx:
            self.weights[key] = self.weights[key] - (y_pred - y) * alpha * val
        # self.weights = self.weights - (y_pred - y) * alpha * f_x


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feat_extractor):
        SentimentClassifier.__init__(self)
        self.feat_extractor = feat_extractor
        self.indexer = self.feat_extractor.get_indexer()
        self.vocab_size = self.indexer.__len__()
        self.weights = np.zeros((self.vocab_size,))
        self.feature_dict = {}

    def get_feature(self, ex_words: List[str]) -> List[int]:
        ex_sent = ''.join(ex_words)
        if ex_sent not in self.feature_dict:
            f_x = self.feat_extractor.extract_features(ex_words)
            # for key, val in f_x:
            #     key = self.indexer.index_of(key)

            self.feature_dict[ex_sent] = f_x
        else:
            f_x = self.feature_dict[ex_sent]
        return f_x

    def predict(self, ex_words: List[str]) -> int:

        fx = self.get_feature(ex_words)

        wfx = 0.
        for key,val in fx:
            wfx += self.weights[key] * val
        
        p = sigmoid(wfx)

        y_ret = 1 if p >0.5 else 0
        return y_ret
    
    def update(self, ex_words, y, y_pred, alpha):
        fx = self.get_feature(ex_words)

        wfx = 0.
        for key,val in fx:
            wfx += self.weights[key] * val
        
        p = sigmoid(wfx)

        # if y==1: # y_pred = 0
        #     # self.weights = self.weights - alpha * f_x * (p-1)
        #     for key,val in fx:
        #         self.weights[key] = self.weights[key] + alpha*val*(1-p)
        # elif y==0: # y_pred = 1
        #     # self.weights = self.weights - alpha * f_x * p
        #     for key,val in fx:
        #         self.weights[key] = self.weights[key] - alpha*val*(p)
        
        '''
        We can combine to update rules into one line 
        p = log(ex/(1+ex)) = x - log(1+ex)
        dp/dx = 1 - ex/(1+ex) = 1/(1+ex)
        loss = -y*log(p)-(1-y)*log(1-p)
        d(loss)/dw = -y*fx*(1-p)+(1-y)*fx*p
        '''
        for key,val in fx:
                self.weights[key] = self.weights[key] - alpha * (-y*val*(1-p)+(1-y)*val*p)
    
    def total_loss(self, train_exs):
        loss_sum=0
        for ex in train_exs:
            y = ex.label
            x = ex.words
            fx = self.get_feature(ex.words)

            wfx = 0.
            for key,val in fx:
                wfx += self.weights[key] * val
            
            p = sigmoid(wfx)
            loss = - y * np.log(p) - (1 - y) * np.log(1 - p) 
            loss_sum+=loss
        
        return loss_sum / float(len(train_exs)) # normalize 


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    # extract features in advance
    for ex in train_exs:
        feat_extractor.add_features(ex.words)
    # for IDF
    try:
        feat_extractor.idf_extractor(train_exs)
    except:
        pass
    
    model = PerceptronClassifier(feat_extractor)
    epochs = 20
    alpha = 1
    for t in tqdm(range(epochs)):
        # shuffle data & sample
        random.shuffle(train_exs)
        sample_size = int((len(train_exs)))
        sampled_exs = train_exs[:sample_size]
        
        for ex in sampled_exs:
            y = ex.label
            y_pred = model.predict(ex.words)
            model.update(ex.words, y, y_pred, alpha)
            # f_x = feat_extractor.extract_features(ex.words)
            # model.weights = [model.weights[i] - (y_pred - y) * alpha for i in f_x]
        
        alpha = alpha *0.8
        # alpha = alpha / (t+1)
    
    return model


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # extract features in advance
    for ex in train_exs:
        feat_extractor.add_features(ex.words)
    # for IDF
    try:
        feat_extractor.idf_extractor(train_exs)
    except:
        pass

    model = LogisticRegressionClassifier(feat_extractor)
    epochs = 30
    if(isinstance(feat_extractor,BetterFeatureExtractor)):
        alpha = 1.
    else:
        alpha = 0.5
    for t in tqdm(range(epochs)):
        if(isinstance(feat_extractor,BetterFeatureExtractor)):
            alpha = alpha / (t+1)
        # print(alpha)
        # alpha = alpha *0.9

        # shuffle data & sample
        random.shuffle(train_exs)
        sample_size = int((len(train_exs)))
        sampled_exs = train_exs[:sample_size]
        
        for ex in sampled_exs:
            # print(feat_extractor.extract_features(ex.words))
            y = ex.label
            y_pred = model.predict(ex.words)
            model.update(ex.words, y, y_pred, alpha)
            # print(ex, y, y_pred)
        
    return model


def train_model(args, train_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model