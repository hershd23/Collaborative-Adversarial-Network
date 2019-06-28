from re import sub
import torch
import csv
import itertools
import random
from random import shuffle
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split as split_data
import numpy as np

class Data(object):
    def __init__(self, data_train_file, data_test_file, train_ratio=0.8, max_len=None,
                 vocab_limit=None, sentence_cols=None, score_col=None):
        self.data_train_file = data_train_file
        self.data_test_file = data_test_file
        self.train_ratio = train_ratio
        self.max_len = max_len
        self.vocab_size = 1
        self.vocab_limit = vocab_limit

        
        self.score_col = 'is_paraphrase'
        self.sequence_cols = ['sentence_A', 'sentence_B']

        self.x_train = list()
        self.y_train = list()
        self.x_val = list()
        self.y_val = list()
        self.x_test = list()
        self.y_test = list()
        self.vocab = set('PAD')
        self.word2index = {'PAD':0}
        self.index2word = {0:'PAD'}
        self.word2count = dict()

        self.use_cuda = torch.cuda.is_available()
        self.run()

    def text_to_word_list(self, text):
        ''' Pre process and convert texts to a list of words '''
        text = str(text)
        text = text.lower()

        # Clean the text
        text = sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = sub(r"what's", "what is ", text)
        text = sub(r"\'s", " ", text)
        text = sub(r"\'ve", " have ", text)
        text = sub(r"can't", "cannot ", text)
        text = sub(r"n't", " not ", text)
        text = sub(r"i'm", "i am ", text)
        text = sub(r"\'re", " are ", text)
        text = sub(r"\'d", " would ", text)
        text = sub(r"\'ll", " will ", text)
        text = sub(r",", " ", text)
        text = sub(r"\.", " ", text)
        text = sub(r"!", " ! ", text)
        text = sub(r"\/", " ", text)
        text = sub(r"\^", " ^ ", text)
        text = sub(r"\+", " + ", text)
        text = sub(r"\-", " - ", text)
        text = sub(r"\=", " = ", text)
        text = sub(r"'", " ", text)
        text = sub(r"(\d+)(k)", r"\g<1>000", text)
        text = sub(r":", " : ", text)
        text = sub(r" e g ", " eg ", text)
        text = sub(r" b g ", " bg ", text)
        text = sub(r" u s ", " american ", text)
        text = sub(r"\0s", "0", text)
        text = sub(r" 9 11 ", "911", text)
        text = sub(r"e - mail", "email", text)
        text = sub(r"j k", "jk", text)
        text = sub(r"\s{2,}", " ", text)

        text = text.split()

        return text

    def load_data(self):
        stops = set(stopwords.words('english'))

        # Load data set
        data_df = pd.read_csv(self.data_train_file, sep='\t')
        data_test_df = pd.read_csv(self.data_test_file, sep='\t')

        # Iterate over required sequences of provided dataset
        for index, row in data_df.iterrows():
            # Iterate through the text of both questions of the row
            for sequence in self.sequence_cols:
                s2n = []  # Sequences with words replaces with indices
                for word in self.text_to_word_list(row[sequence]):
                    # Remove unwanted words
                    
                    if word in stops:
                        continue

                    if word not in self.vocab:
                        self.vocab.add(word)
                        self.word2index[word] = self.vocab_size
                        self.word2count[word] = 1
                        s2n.append(self.vocab_size)
                        self.index2word[self.vocab_size] = word
                        self.vocab_size += 1
                    else:
                        self.word2count[word] += 1
                        s2n.append(self.word2index[word])

                # Replace |sequence as word| with |sequence as number| representation
                data_df.at[index, sequence] = s2n


        for index, row in data_test_df.iterrows():
            # Iterate through the text of both questions of the row
            for sequence in self.sequence_cols:
                s2n = []  # Sequences with words replaces with indices
                for word in self.text_to_word_list(row[sequence]):
                    # Remove unwanted words
                    
                    if word in stops:
                        continue

                    if word not in self.vocab:
                        self.vocab.add(word)
                        self.word2index[word] = self.vocab_size
                        self.word2count[word] = 1
                        s2n.append(self.vocab_size)
                        self.index2word[self.vocab_size] = word
                        self.vocab_size += 1
                    else:
                        self.word2count[word] += 1
                        s2n.append(self.word2index[word])

                # Replace |sequence as word| with |sequence as number| representation
                data_test_df.at[index, sequence] = s2n

        return data_df, data_test_df

    def convert_to_tensors(self):
        for data in [self.x_train, self.x_val, self.x_test]:
            for i, pair in enumerate(data):
                data[i][0] = torch.LongTensor(data[i][0])
                data[i][1] = torch.LongTensor(data[i][1])

                if self.use_cuda:
                    data[i][0] = data[i][0].cuda()
                    data[i][1] = data[i][1].cuda()

        self.y_train = torch.FloatTensor(self.y_train)
        self.y_val = torch.FloatTensor(self.y_val)
        self.y_test = torch.FloatTensor(self.y_test)

        if self.use_cuda:
            self.y_train = self.y_train.cuda()
            self.y_val = self.y_val.cuda()
            self.y_test = self.y_test.cuda()

    def run(self):
        # Loading data and building vocabulary.
        data_df, data_test_df = self.load_data()
        data_size = len(data_df)

        X = data_df[self.sequence_cols]
        Y = data_df[self.score_col]

        self.x_test = data_test_df[self.sequence_cols]
        self.y_test = data_test_df[self.score_col]

        self.x_train, self.x_val, self.y_train, self.y_val = split_data(X, Y, train_size=self.train_ratio)

        # Convert labels to their numpy representations
        self.y_train = self.y_train.values
        self.y_val = self.y_val.values
        self.y_test = self.y_test.values

        training_pairs = []
        training_scores = []
        validation_pairs = []
        validation_scores = []
        test_pairs = []
        test_scores = []

        # Split to lists
        i = 0
        for index, row in self.x_train.iterrows():
            sequence_1 = row[self.sequence_cols[0]]
            sequence_2 = row[self.sequence_cols[1]]
            if len(sequence_1) > 0 and len(sequence_2) > 0:
                training_pairs.append([sequence_1, sequence_2])
                training_scores.append(float(self.y_train[i]))
            i += 1
        self.x_train = training_pairs
        self.y_train = training_scores

        print('Number of Training Samples   :', len(training_scores))
        #print('Number of Training Negative Samples   :', len(training_scores) - sum(training_scores))

        i = 0
        for index, row in self.x_val.iterrows():
            sequence_1 = row[self.sequence_cols[0]]
            sequence_2 = row[self.sequence_cols[1]]
            if len(sequence_1) > 0 and len(sequence_2) > 0:
                validation_pairs.append([sequence_1, sequence_2])
                validation_scores.append(float(self.y_val[i]))
            i += 1

        self.x_val = validation_pairs
        self.y_val = validation_scores

        i = 0
        for index, row in self.x_test.iterrows():
            sequence_1 = row[self.sequence_cols[0]]
            sequence_2 = row[self.sequence_cols[1]]
            if len(sequence_1) > 0 and len(sequence_2) > 0:
                test_pairs.append([sequence_1, sequence_2])
                test_scores.append(float(self.y_test[i]))
            i += 1

        self.x_test = test_pairs
        self.y_test = test_scores

        print('Number of Validation Samples   :', len(validation_scores))
        #print('Number of Validation Negative Samples   :', len(validation_scores) - sum(validation_scores))

        assert len(self.x_train) == len(self.y_train)
        assert len(self.x_val) == len(self.y_val)
        assert len(self.x_test) == len(self.y_test)

        self.convert_to_tensors()