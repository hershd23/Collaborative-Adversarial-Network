import torch
import pandas as pd 

#IMPORTING LIBRARIES

from data import Data
from sklearn.metrics import f1_score, average_precision_score
import sklearn

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import numpy as np

use_cuda = torch.cuda.is_available()

data_file = "./dataset/train.tsv"
data_test_file = "./dataset/test.tsv"
training_ratio = 0.9
max_len = 30
tracking_pair = False
hidden_size = 50
batch_size = 1
num_iters = 50
learning_rate = 0.03

"""# DATA"""

data = Data(data_file,data_test_file,training_ratio,max_len)

print(len(data.word2index))

"""# Embeddings"""

embd_file = "./glove-global-vectors-for-word-representation/glove.6B.100d.txt"

from embedding_helper2 import Get_Embedding

embedding = Get_Embedding(embd_file, data.word2index)
embedding_size = embedding.embedding_matrix.shape[1]

print(embedding_size)

len(embedding.embedding_matrix)

import torch.nn as nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F

def commonWords(sen_1, sen_2):
  d = np.empty(len(data.word2index), dtype=int)
  for i in range(len(d)):
    d[i] = -1
    
  flag = False
    
  listPairs = []
  list1 = []
  list2 = []
  for i in range(len(sen_1)):
    d[sen_1[i]] = i
    
  for i in range(len(sen_2)):
    if d[sen_2[i]] > 1 and sen_2[i] > 0 :
      list1.append(d[sen_2[i]])
      list2.append(i)
      flag = True
      
    
  list1 = list(dict.fromkeys(list1))
  list2 = list(dict.fromkeys(list2))
  
  listPairs.append(list1)
  listPairs.append(list2)
  return listPairs

def max_pool(e_list):
  e_list = np.array(e_list)
  
  for i in range(len(e_list)):
    e_list[i] = e_list[i].data.cpu().numpy()
  mp = []
  for i in range(100):
    m = e_list[0][i]
    for j in range(len(e_list)):
      m = max(m, e_list[j][i])
    mp.append(m)
      
  #print("Length of mp = " + str(len(mp)))
  return torch.cuda.FloatTensor(mp)

"""# GAN MODEL"""

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
    nn.init.xavier_uniform_(m.weight.data).cuda()
    nn.init.constant_(m.bias.data, 0).cuda()

learning_rate_G = 0.03
learning_rate_D = 0.03

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.main = nn.Sequential(
            nn.Linear(100, 100),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input).cuda()

netG = Generator()
if use_cuda: netG = netG.cuda()
netG.apply(weights_init)

class Discriminator(nn.Module):
    def __init__(self):
      super(Discriminator, self).__init__()
      self.use_cuda = torch.cuda.is_available()
      self.main = nn.Sequential(
        nn.Linear(100, 2),
        nn.Softmax(dim = 1)
      ) 
      
    def forward(self, input):
      return self.main(input).cuda()

netD = Discriminator()
if use_cuda: netD = netD.cuda()
netD.apply(weights_init)

real_label = torch.tensor([0,1])
fake_label = torch.tensor([1,0])
optimizerD = optim.Adadelta(netD.parameters(), lr=learning_rate_G)
optimizerG = optim.Adadelta(netG.parameters(), lr=learning_rate_D)

class Dropout_layer(nn.Module):
  def __init__(self):
    super(Dropout_layer, self).__init__()
    self.d = nn.Dropout(p=0.5)
    
  def forward(self, input):
    return self.d(input).cuda()

dropout_layer = Dropout_layer()
if use_cuda: dropout_layer = dropout_layer.cuda()

class Final_layer(nn.Module):
    def __init__(self):
        super(Final_layer, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.main = nn.Sequential(
            nn.Linear(101, 2),
            nn.Softmax(dim = 1)
        )
    def forward(self, input):
        return self.main(input).cuda()

net_final = Final_layer()
if use_cuda: net_final = net_final.cuda()
net_final.apply(weights_init)

final_par = list(net_final.parameters())

"""# MALSTM MODEL"""

class Manhattan_LSTM(nn.Module):
    def __init__(self, hidden_size, embedding, train_embedding = False):
        super(Manhattan_LSTM, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
        self.embedding.weight = nn.Parameter(embedding)
        self.input_size = embedding.shape[1]
        
        self.embedding.weight.requires_grad = train_embedding
        
        self.lstm_1 = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, bidirectional=True)
        self.lstm_2 = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, bidirectional=True)
        
    def exponent_neg_manhattan_distance(self, x1, x2):
        return torch.exp(-torch.sum(torch.abs(x1 - x2), dim=1))
    
    def forward(self, input, hidden):
        
        #print(input[0])
        #print(input[1])
        
        ip0 = input[0].t()
        ip1 = input[1].t()
        
        commonList = []
        
        for i in range(batch_size):
            listPairs = commonWords(ip0[i], ip1[i])
            commonList.append(listPairs)
    
        commonList = np.array(commonList)
        
        #print(commonList)
        input_len = len(input[1])
        
        embedded_1 = self.embedding(input[0])
        embedded_2 = self.embedding(input[1])
        
        bs = embedded_1.size()[1]
        outputs_1, hidden_1 = self.lstm_1(embedded_1, hidden)
        outputs_2, hidden_2 = self.lstm_1(embedded_2, hidden)
        
        max_pool_1 = F.adaptive_max_pool1d(outputs_1.permute(1,2,0),1).view(batch_size,-1)
        max_pool_2 = F.adaptive_max_pool1d(outputs_2.permute(1,2,0),1).view(batch_size,-1)
        
        att_weights = torch.bmm(max_pool_1.view(batch_size, 1, 100), outputs_2.view(batch_size, 100, input_len)).view(batch_size, input_len)
        
        att_softmax = torch.zeros([batch_size, input_len])
        for i in range(batch_size):
          att_softmax[i] = F.softmax(att_weights[i], dim = 0)
        
        new_pool = torch.bmm(att_softmax.view(batch_size, 1, input_len), outputs_2.view(batch_size, input_len, 100).cpu()).view(batch_size, 100).cuda()
        
        ehs_1 = []
        for i in range(batch_size):
            e_list = []
            for j in range(len(commonList[i][0])):
                x = commonList[i][0][j]
              
                e_list.append(outputs_1[x][i])
            if len(e_list) > 0:
                mp1 = max_pool(e_list)
            else:
                mp1 = torch.zeros(100)
              
            ehs_1.append(mp1.cuda())
        
        
        ehs_2 = []
        for i in range(batch_size):
            e_list = []
            for j in range(len(commonList[i][1])):
                x = commonList[i][1][j]
              
                e_list.append(outputs_2[x][i])
            if len(e_list) > 0:
                mp2 = max_pool(e_list)
            else:
                mp2 = torch.zeros(100)
              
            ehs_2.append(mp2.cuda())
            
        elitehs_1 = torch.zeros(batch_size, 100)
        for i in range(batch_size):
            elitehs_1[i] = ehs_1[i]
          
        elitehs_2 = torch.zeros(batch_size, 100)
        for i in range(batch_size):
            elitehs_2[i] = ehs_2[i]
        
        elitehs_1.cuda()
        elitehs_2.cuda()
        #similarity_scores = self.exponent_neg_manhattan_distance(ths_1.cuda(), ths_2.cuda())
        similarity_scores = self.exponent_neg_manhattan_distance(max_pool_1, new_pool)
        
        return similarity_scores, elitehs_1, elitehs_2
    
    def init_weights(self):
        for name_1, param_1 in self.lstm_1.named_parameters():
            if 'bias' in name_1:
                nn.init.constant_(param_1, 0.01)
            elif 'weight' in name_1:
                nn.init.xavier_uniform_(param_1)

        lstm_1 = self.lstm_1.state_dict()
        lstm_2 = self.lstm_2.state_dict()

        for name_1, param_1 in lstm_1.items():
            # Backwards compatibility for serialized parameters.
            if isinstance(param_1, torch.nn.Parameter):
                param_1 = param_1.data

            lstm_2[name_1].copy_(param_1)

    def init_hidden(self, batch_size):
        # Hidden dimensionality : 2 (h_0, c_0) x Num. Layers * Num. Directions x Batch Size x Hidden Size
        result = torch.zeros(2, 2, batch_size, self.hidden_size)
        result = tuple(result)

        if self.use_cuda: 
            result = (result[0].cuda(), result[1].cuda())
            return result
        else: return result

model = Manhattan_LSTM(hidden_size, embedding.embedding_matrix, train_embedding=False)
if use_cuda: model = model.cuda()
model.init_weights()

import time
import random
from torch import optim
import torch.nn.utils.rnn as rnn

x_train = data.x_train
x_val = data.x_val
y_train = data.y_train
y_val = data.y_val
x_test = data.x_test
y_test = data.y_test
train_samples = len(x_train)
val_samples = len(x_val)
test_samples = len(x_test)
test_samples

criterion = nn.BCELoss()
print_every = 1
print_loss_total = 0.0
train_loss = 0.0
max_acc = 0.7
par = 0.5

model_trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
model_trainable_parameters = model_trainable_parameters + final_par
model_trainable_parameters = tuple(model_trainable_parameters)
model_optimizer = optim.Adam(model_trainable_parameters, lr=learning_rate)

from helper import Helper
help_fn = Helper()

#run to load the base model

model.load_state_dict(torch.load("../model_weights/model_weights.pt"))
model.eval()
model.train()


start = time.time()
print('Beginning Model Training.\n')
batch_size = 16

for epoch in range(0, num_iters):
    model_loss1 = 0.0
    gen_loss1 = 0.0
    dis_loss1 = 0.0
    fin_loss1 = 0.0
    train_loss1 = 0.0
    model_loss2 = 0.0
    gen_loss2 = 0.0
    dis_loss2 = 0.0
    fin_loss2 = 0.0
    train_loss2 = 0.0
    val_loss = 0.0
    for i in range(0, train_samples, batch_size):
        input_variables = x_train[i:i+batch_size]
        similarity_scores = y_train[i:i+batch_size]
        
        sequences_1 = [sequence[0] for sequence in input_variables]
        sequences_2 = [sequence[1] for sequence in input_variables]
        batch_size = len(sequences_1)
        
        # Make a tensor for the similarity scores
        
        sim_scores_2d = torch.zeros([batch_size, 2])
        for j in range(batch_size):
          if similarity_scores[j] == 0:
            sim_scores_2d[j] = fake_label
          else:
            sim_scores_2d[j] = real_label
            
        sim_scores_2d = sim_scores_2d.cuda()

        temp = rnn.pad_sequence(sequences_1 + sequences_2)
        sequences_1 = temp[:, :batch_size]
        sequences_2 = temp[:, batch_size:]

        model_optimizer.zero_grad()
        loss_s = 0.0
        
        optimizerG.zero_grad()
        loss_g= 0.0
        
        optimizerD.zero_grad()
        loss_d= 0.0

        loss_f = 0.0

        # Initialise the hidden state and pass through the maLSTM
        hidden = model.init_hidden(batch_size)
        output_scores, ehs1, ehs2 = model([sequences_1, sequences_2], hidden)
        
        output_scores = output_scores.view(-1)
        
        loss_s += criterion(output_scores, similarity_scores)
        
        ehs1 = ehs1.cuda()
        ehs2 = ehs2.cuda()
        
        
        # Generator
        gen_feature = netG(ehs2)
        
        # 1. Discriminator for the real class
        discrimm_classes = netD(ehs1)
        labels = torch.zeros(batch_size, 2)
        for j in range(batch_size):
          labels[j] = real_label
          
        labels = labels.cuda()
        
        loss_d += criterion(discrimm_classes, labels)
        
        
        # 2. Discriminator for the fake class
        discrimm_classes = netD(gen_feature)
        labels = torch.zeros(batch_size, 2)
        for j in range(batch_size):
          labels[j] = fake_label
          
        labels = labels.cuda()
          
        loss_d += criterion(discrimm_classes, labels)
        
        #print(discrimm_classes)
        
        # Update generator loss
        loss_g += criterion(discrimm_classes, sim_scores_2d)
        
        d_feature = dropout_layer(gen_feature)
        
        cat_feature = torch.zeros([batch_size, len(d_feature[0])+1])
        for j in range(batch_size):
          for k in range(100):
            cat_feature[j][k] = d_feature[j][k]
          cat_feature[j][100] = output_scores[j]
          
        
        cat_feature = cat_feature.cuda()
        
        final_labels = net_final(cat_feature)
        
        loss_f += criterion(final_labels, sim_scores_2d)
        
        com_loss = (0.6*loss_g) + loss_f + loss_d
        
        com_loss.backward()
        
        model_optimizer.step()
        optimizerG.step()
        
        
        fin_loss1 += loss_f
        model_loss1 += loss_s
        gen_loss1 += loss_g
        dis_loss1 += loss_d
    
        train_loss1 += com_loss
        
        
    for i in range(0, train_samples, batch_size):
        input_variables = x_train[i:i+batch_size]
        similarity_scores = y_train[i:i+batch_size]
        
        sequences_1 = [sequence[0] for sequence in input_variables]
        sequences_2 = [sequence[1] for sequence in input_variables]
        batch_size = len(sequences_1)
        
        # Make a tensor for the similarity scores
        
        sim_scores_2d = torch.zeros([batch_size, 2])
        for j in range(batch_size):
          if similarity_scores[j] == 0:
            sim_scores_2d[j] = fake_label
          else:
            sim_scores_2d[j] = real_label
            
        sim_scores_2d = sim_scores_2d.cuda()

        temp = rnn.pad_sequence(sequences_1 + sequences_2)
        sequences_1 = temp[:, :batch_size]
        sequences_2 = temp[:, batch_size:]
        
        optimizerD.zero_grad()
        loss_d = 0.0
        
        model_optimizer.zero_grad()
        loss_s = 0.0
        
        optimizerG.zero_grad()
        loss_g= 0.0
        
        loss_f = 0.0

        # Initialise the hidden state and pass through the maLSTM
        hidden = model.init_hidden(batch_size)
        output_scores, ehs1, ehs2 = model([sequences_1, sequences_2], hidden)
        
        output_scores = output_scores.view(-1)
        
        loss_s += criterion(output_scores, similarity_scores)
        
        ehs1 = ehs1.cuda()
        ehs2 = ehs2.cuda()
        
        
        # Generator
        gen_feature = netG(ehs2)
        
        # 1. Discriminator for the real class
        discrimm_classes = netD(ehs1)
        labels = torch.zeros(batch_size, 2)
        for j in range(batch_size):
          labels[j] = real_label
          
        labels = labels.cuda()
        
        loss_d += criterion(discrimm_classes, labels)
        
        
        # 2. Discriminator for the fake class
        discrimm_classes = netD(gen_feature)
        labels = torch.zeros(batch_size, 2)
        for j in range(batch_size):
          labels[j] = fake_label
          
        labels = labels.cuda()
          
        loss_d += criterion(discrimm_classes, labels)
        
        loss_g += criterion(discrimm_classes, sim_scores_2d)
        
        d_feature = dropout_layer(gen_feature)
        
        cat_feature = torch.zeros([batch_size, len(d_feature[0])+1])
        for j in range(batch_size):
          for k in range(100):
            cat_feature[j][k] = d_feature[j][k]
          cat_feature[j][100] = output_scores[j]
          
        
        cat_feature = cat_feature.cuda()
        
        final_labels = net_final(cat_feature)
      
        loss_f += criterion(final_labels, sim_scores_2d)
        com_loss = (0.6*loss_g) + loss_f + loss_d
        
        com_loss.backward()
        
        optimizerD.step()
        
        
        fin_loss2 += loss_f
        model_loss2 += loss_s
        gen_loss2 += loss_g
        dis_loss2 += loss_d
    
        train_loss2 += com_loss

     
        
        
    '''if epoch % 5:
        learning_rate *= 0.5
        model_optimizer = optim.Adam(model_trainable_parameters, lr=learning_rate)
        optimizer_final = optim.Adam(net_final.parameters(), lr = learning_rate)
        optimizerD = optim.Adam(netD.parameters(), lr=learning_rate)
        optimizerG = optim.Adam(netG.parameters(), lr=learning_rate)
        '''
    
    
    a_scores = []
    p_scores = []
    corr = 0
    fin_lossv = 0.0
    model_lossv = 0.0
    gen_lossv = 0.0
    dis_lossv = 0.0
    for i in range(0, test_samples, batch_size):
        input_variables = x_test[i:i+batch_size]
        actual_scores = y_test[i:i+batch_size]

        sequences_1 = [sequence[0] for sequence in input_variables]
        sequences_2 = [sequence[1] for sequence in input_variables]
        batch_size = len(sequences_1)
        
        sim_scores_2d = torch.zeros([batch_size, 2])
        for j in range(batch_size):
          if actual_scores[j] == 0:
            sim_scores_2d[j] = fake_label
          else:
            sim_scores_2d[j] = real_label
            
        sim_scores_2d = sim_scores_2d.cuda()

        temp = rnn.pad_sequence(sequences_1 + sequences_2)
        sequences_1 = temp[:, :batch_size]
        sequences_2 = temp[:, batch_size:]

        loss = 0.0
        loss_d = 0.0
        loss_g = 0.0
        loss_f = 0.0
        loss_s = 0.0
        
        hidden = model.init_hidden(batch_size)
        output_scores, ehs1, ehs2 = model([sequences_1, sequences_2], hidden)
        
        output_scores = output_scores.view(-1)
        
        loss_s += criterion(output_scores, actual_scores)
        
        ehs1 = ehs1.cuda()
        ehs2 = ehs2.cuda() 
        gen_feature = netG(ehs2)
        
        # 1. Discriminator for the real class
        discrimm_classes = netD(ehs1)
        labels = torch.zeros(batch_size, 2)
        for j in range(batch_size):
          labels[j] = real_label
          
        labels = labels.cuda()
        
        loss_d += criterion(discrimm_classes, labels)
        
        
        # 2. Discriminator for the fake class
        discrimm_classes = netD(gen_feature)
        labels = torch.zeros(batch_size, 2)
        for j in range(batch_size):
          labels[j] = fake_label
          
        labels = labels.cuda()
          
        loss_d += criterion(discrimm_classes, labels)
        
        loss_g += criterion(discrimm_classes, sim_scores_2d)
        
        d_feature = dropout_layer(gen_feature)
        
        cat_feature = torch.zeros([batch_size, len(d_feature[0])+1])
        for j in range(batch_size):
          for k in range(100):
            cat_feature[j][k] = d_feature[j][k]
          cat_feature[j][100] = output_scores[j]
          
        cat_feature = cat_feature.cuda()
        
        final_labels = net_final(cat_feature)
        
        loss_f += criterion(final_labels, sim_scores_2d)
        
        loss = loss_f + loss_d + (0.6*loss_g)
        
        fin_lossv += loss_f
        model_lossv += loss_s
        gen_lossv += loss_g
        dis_lossv += loss_d
        
        val_loss += loss
        
        for j in range(0, batch_size):
          acts = actual_scores[j].data.cpu().numpy()
          preds = final_labels[j].data.cpu().numpy()
          a_scores.append(acts)

          if preds[0] >= 0.5 and acts == 0:
            corr = corr+1
            p_scores.append(0)
          elif preds[1] >= 0.5 and acts == 1:
            corr = corr+1
            p_scores.append(1)
          elif preds[0] >=0.5:
            p_scores.append(0)
          else:
            p_scores.append(1)
          
    
    if epoch % print_every == 0:
        print('%s (%d)' % (help_fn.time_slice(start, (epoch+1) / num_iters), epoch))
        print("Train Loss    " + str(train_loss2.data.cpu().numpy()) + "    Val loss    " + str(val_loss.data.cpu().numpy()))
        print("LSTM loss 1   " + str(model_loss1.data.cpu().numpy()) + "    Gen loss 1   " + str(gen_loss1.data.cpu().numpy()) + "    Dis loss 1   " + str(dis_loss1.data.cpu().numpy()) + "    Fin loss 1   " + str(fin_loss1.data.cpu().numpy()))
        print("LSTM loss 2   " + str(model_loss2.data.cpu().numpy()) + "    Gen loss 2   " + str(gen_loss2.data.cpu().numpy()) + "    Dis loss 2   " + str(dis_loss2.data.cpu().numpy()) + "    Fin loss 2   " + str(fin_loss2.data.cpu().numpy()))
        print("LSTM loss v   " + str(model_lossv.data.cpu().numpy()) + "    Gen loss v   " + str(gen_lossv.data.cpu().numpy()) + "    Dis loss v   " + str(dis_lossv.data.cpu().numpy()) + "    Fin loss v   " + str(fin_lossv.data.cpu().numpy()))
        print(" Test Accuracy    " + str(corr/len(a_scores)) + "    f1 score    " + str(f1_score(p_scores, a_scores)))
        
        acc = corr/len(a_scores)
        
        if acc > max_acc :
          max_acc = acc
          torch.save(model.state_dict(), "../model_weights/model_weights.pt")
          torch.save(netG.state_dict(), "../model_weights/netG_weights.pt" )
          torch.save(netD.state_dict(), "../model_weights/netD_weights.pt")
          torch.save(net_final.state_dict(), "../model_weights/netfinal_weights.pt")
          print("Model Saved!")