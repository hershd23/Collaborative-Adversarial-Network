import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import torch

class Get_Embedding(object):
    def __init__(self, file_path, word_index):
        self.use_cuda = torch.cuda.is_available()
        self.embedding_size = 100 # Dimensionality of Glove
        self.vocab = []
        self.word_vec = []
        self.embedding_matrix = self.create_embed_matrix(file_path, word_index)


    def findWord(self, word):
        for i in range(len(self.vocab)):
            if word == self.vocab[i]:
                return i
        
        return -1

    def create_embed_matrix(self, file_path, word_index):
        f = open("../embeddings/paragram.txt", encoding = "latin1")

        for x in f:
            data = x.split("\t")
            data[-1] = data[-1][0:-1]
            embed = np.array(data[1:], dtype = np.float32)
            self.vocab.append(data[0])
            self.word_vec.append(embed)

        print(type(self.word_vec[0]))
        para_matrix = np.zeros((len(word_index)+1, 25))
        for word, i in word_index.items():
            j = self.findWord(word)
            if j == -1:
                continue
            para_matrix[i] = self.word_vec[j]

        print("Paragrams done")

        glove2word2vec(glove_input_file=file_path, word2vec_output_file="gensim_glove_vectors.txt")
        glove = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)

        # Prepare Embedding Matrix.
        embedding_matrix = np.zeros((len(word_index)+1, self.embedding_size))
        for word, i in word_index.items():
            # words not found in embedding index will be all-zeros.
            if word not in glove.vocab:
                continue
            embedding_matrix[i] = glove.word_vec(word)

        del glove

        e_matrix = np.concatenate((embedding_matrix, para_matrix), axis = 1)

        e_matrix = torch.FloatTensor(e_matrix)
        if self.use_cuda: e_matrix = e_matrix.cuda()

        return e_matrix
