import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import torch

class Get_Embedding(object):
    def __init__(self, file_path, word_index):
        self.use_cuda = torch.cuda.is_available()
        self.embedding_size = 100 # Dimensionality of Glove
        self.embedding_matrix = self.create_embed_matrix(file_path, word_index)

    def create_embed_matrix(self, file_path, word_index):
        glove2word2vec(glove_input_file=file_path, word2vec_output_file="gensim_glove_vectors.txt")

        from gensim.models.keyedvectors import KeyedVectors
        glove = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)

        # Prepare Embedding Matrix.
        embedding_matrix = np.zeros((len(word_index)+1, self.embedding_size))

        for word, i in word_index.items():
            # words not found in embedding index will be all-zeros.
            if word not in glove.vocab:
                continue
            embedding_matrix[i] = glove.word_vec(word)

        del glove

        embedding_matrix = torch.FloatTensor(embedding_matrix)
        if self.use_cuda: embedding_matrix = embedding_matrix.cuda()

        return embedding_matrix
