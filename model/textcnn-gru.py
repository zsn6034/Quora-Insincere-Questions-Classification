import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

import math
import datetime
import string, re

#define some help function
def cuda_available(tensor):
    if torch.cuda.is_available:
        return tensor.cuda()
    return tensor
def changeTime(allTime):  
    day = 24*60*60  
    hour = 60*60  
    min = 60  
    if allTime <60:          
        return  "%d sec"%math.ceil(allTime)  
    elif  allTime > day:  
        days = divmod(allTime,day)   
        return "%d days, %s"%(int(days[0]),changeTime(days[1]))  
    elif allTime > hour:  
        hours = divmod(allTime,hour)  
        return '%d hours, %s'%(int(hours[0]),changeTime(hours[1]))  
    else:  
        mins = divmod(allTime,min)  
        return "%d mins, %d sec"%(int(mins[0]),math.ceil(mins[1]))
    
def clean(text): 
    ## Remove puncuation
    text = text.translate(string.punctuation)
    ## Convert words to lower case and split them
    text = text.lower()
    ## Remove stop words
    #text = text.split()
    #stops = set(stopwords.words("english"))
    #text = [w for w in text if not w in stops and len(w) >= 3]
    #text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub('[^a-zA-Z]',' ', text)
    text = re.sub('  +',' ',text)
    #text = text.split()
    #stemmer = SnowballStemmer('english')
    #stemmed_words = [stemmer.stem(word) for word in text]
    #text = " ".join(stemmed_words)
    return text
    
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
submission = pd.read_csv('../input/sample_submission.csv')
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)
print("submission shape :", submission.shape)

train_df['clean_text'] = train_df['question_text'].apply(clean)
test_df['clean_text'] = test_df['question_text'].apply(clean)

#some config values
embedding_size = 300
hidden_size = 256
words_size = 95000   
max_len = 70  
lr = 0.0001  
batch_size = 48   
epoch = 3  
dropout = 0.1
num_filters = 60

#split to train and val
#train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=2018)

#fill up the missing values
train_X = train_df['clean_text'].fillna('_na_').values  #1175509
#val_X = val_df['clean_text'].fillna('_na_').values  #130613
test_X = test_df['clean_text'].fillna('_na_').values  #56370

#Tokenize the sequences
tokenizer = Tokenizer(num_words=words_size)
tokenizer.fit_on_texts(list(train_X) + list(test_X))
train_X = tokenizer.texts_to_sequences(train_X)
#val_X =tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

#Pad the sequences
train_X = pad_sequences(train_X, maxlen=max_len)
#val_X = pad_sequences(val_X, maxlen=max_len)
test_X = pad_sequences(test_X, maxlen=max_len)

#Get the target values
train_y = train_df['target'].values  #1175509
#val_y = val_df['target'].values  #130613

#numpy2tensor
tensor_X = torch.from_numpy(train_X)
tensor_y = torch.from_numpy(train_y)

#build a train-loader to collect train data
quoraTrainSet = TensorDataset(tensor_X, tensor_y)
train_loader = DataLoader(dataset=quoraTrainSet,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=2)
                         
def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    nb_words = min(words_size, len(word_index))
    embedding_matrix = np.zeros((nb_words, embedding_size))
    for word, i in word_index.items():
        if i >= words_size: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix 
    
def load_fasttext(word_index):    
    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)
    nb_words = min(words_size, len(word_index))
    embedding_matrix = np.zeros((nb_words, embedding_size))
    for word, i in word_index.items():
        if i >= words_size: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix 

def load_para(word_index):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)
    nb_words = min(words_size, len(word_index))
    embedding_matrix = np.zeros((nb_words, embedding_size))
    for word, i in word_index.items():
        if i >= words_size: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix 
    
embedding_matrix = load_glove(tokenizer.word_index)
#embedding_matrix_2 = load_fasttext(tokenizer.word_index)
#embedding_matrix_3 = load_para(tokenizer.word_index)
#embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_2, embedding_matrix_3], axis = 0)
np.shape(embedding_matrix)

#use glove
class CNN_Classifier(nn.Module):  #max_len = 50
    def __init__(self, input_size, pretrained_embeddings, embedding_size=300, max_len=70,
                 num_filters=36, dropout=0.1):
        super(CNN_Classifier, self).__init__()
        self.num_filters = num_filters
        self.embedding1 = nn.Embedding(input_size, embedding_size)
        self.embedding2 = nn.Embedding(input_size, embedding_size)
        self.embedding1.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding1.weight.requires_grad = False
        self.embedding2.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.cnn1 = nn.Conv2d(1, self.num_filters, (1, embedding_size), 1)  #60, 70, 1
        self.cnn2 = nn.Conv2d(1, self.num_filters, (2, embedding_size), 1)  #60, 69, 1
        self.cnn3 = nn.Conv2d(1, self.num_filters, (3, embedding_size), 1)  #60, 68, 1        
        self.cnn5 = nn.Conv2d(1, self.num_filters, (5, embedding_size), 1)  #60, 66, 1
        self.max1_pool = nn.MaxPool2d((max_len - 1 + 1, 1))
        self.max2_pool = nn.MaxPool2d((max_len - 2 + 1, 1))
        self.max3_pool = nn.MaxPool2d((max_len - 3 + 1, 1))
        self.max5_pool = nn.MaxPool2d((max_len - 5 + 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.num_filters * 8, 1)
        
    def forward(self, question, train=True):
        batch = question.size(0)
        question_embed1 = self.embedding1(question).unsqueeze(1)  #bacth x 1 x max_len x embedding_size
        question_embed2 = self.embedding2(question).unsqueeze(1)  #bacth x 1 x max_len x embedding_size 
        #bacth x 1 x max_len x (2 x embedding_size)
        out_cnn11 = torch.tanh(self.cnn1(question_embed1))  #batch x 60 x 70 x 1
        out_cnn21 = torch.tanh(self.cnn2(question_embed1))  #batch x 60 x 69 x 1
        out_cnn31 = torch.tanh(self.cnn3(question_embed1))  #batch x 60 x 68 x 1
        out_cnn51 = torch.tanh(self.cnn5(question_embed1))  #batch x 60 x 66 x 1
        out_cnn12 = torch.tanh(self.cnn1(question_embed2))  #batch x 60 x 70 x 1
        out_cnn22 = torch.tanh(self.cnn2(question_embed2))  #batch x 60 x 69 x 1
        out_cnn32 = torch.tanh(self.cnn3(question_embed2))  #batch x 60 x 68 x 1
        out_cnn52 = torch.tanh(self.cnn5(question_embed2))  #batch x 60 x 66 x 1
        out_pool11 = self.max1_pool(out_cnn11).view(batch, -1)  #batch x 60
        out_pool21 = self.max2_pool(out_cnn21).view(batch, -1)  #batch x 60
        out_pool31 = self.max3_pool(out_cnn31).view(batch, -1)  #batch x 60
        out_pool51 = self.max5_pool(out_cnn51).view(batch, -1)  #batch x 60
        out_pool12 = self.max1_pool(out_cnn12).view(batch, -1)  #batch x 60
        out_pool22 = self.max2_pool(out_cnn22).view(batch, -1)  #batch x 60
        out_pool32 = self.max3_pool(out_cnn32).view(batch, -1)  #batch x 60
        out_pool52 = self.max5_pool(out_cnn52).view(batch, -1)  #batch x 60
        aggragate = torch.cat([out_pool11, out_pool21, out_pool31, out_pool51, 
                               out_pool12, out_pool22, out_pool32, out_pool52,], 1)  #batch x 480
        if train:
            aggragate = self.dropout(aggragate)
        return torch.sigmoid(self.linear(aggragate))
        
#use glove
model_cnn = CNN_Classifier(words_size, embedding_matrix, embedding_size=embedding_size, 
                           max_len=max_len, num_filters=num_filters, dropout=dropout).cuda()
criterion = nn.BCELoss() 
#optimizer = torch.optim.Adam(model_cnn.parameters(), lr=lr)  #non-static
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_cnn.parameters()), lr=lr)  #static
print(model_cnn)
print("max_len={0}, num_filters={1}, dropout={2}, batch_size={3}".format(max_len, num_filters, dropout, batch_size))

#train process
train_loss_array = []  #keep total loss
starttime = datetime.datetime.now()
print("Training for %d epochs..." % epoch)
for i in range(epoch):
    train_loss = 0
    for j, (question, label) in enumerate(train_loader):
        optimizer.zero_grad()
        question = cuda_available(question).long()
        label = cuda_available(label)
        output = model_cnn(question, train=True).squeeze(1)
        loss = criterion(output, label.float())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss_array.append(train_loss)
    endtime = datetime.datetime.now()
    print("epoch is %d, train_loss is %.4f, batch is %d, cost time is about %s" % 
          (i, train_loss, batch_size, changeTime((endtime - starttime).seconds)))
print("train finish!")

#use glove
class GRU_Classifier(nn.Module):  #max_len = 70
    def __init__(self, input_size, pretrained_embeddings, embedding_size=300, hidden_size=100,
                 max_len=70, dropout=0.5):
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        super(GRU_Classifier, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        #self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, 
                          bidirectional=True, batch_first=True)
        self.max_pool = nn.MaxPool1d(kernel_size=max_len)
        self.avg_pool = nn.AvgPool1d(kernel_size=max_len)
        #out 
        self.linear1 = nn.Linear(6 * hidden_size, 3 * hidden_size)
        self.linear2 = nn.Linear(3 * hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, question, train=True):
        batch = question.size(0)
        question_embed = self.embedding(question)  #bacth x max_len x embedding_size
        #gru_output: batch x max_len x (2 x hidden_size), hidden: num_directions(2) x batch x hidden_size
        hidden = self.init_hidden(batch)
        gru_output, hidden = self.gru(question_embed, hidden) 
        gru_output = gru_output.transpose(1,2)  #batch x (2 x hidden_size) x max_len
        out_max = self.max_pool(gru_output).view(batch, -1)  #batch x (2 x hidden_size)
        out_avg = self.avg_pool(gru_output).view(batch, -1)  #batch x (2 x hidden_size)
        hidden = hidden.transpose(0, 1).contiguous().view(batch, -1)  #batch x (2 x hidden_size)   
        agg = torch.cat([out_max, out_avg, hidden], 1)  #batch x (6 x hidden_size)   
        if train:
            agg = self.dropout(agg)
        agg = torch.relu(self.linear1(agg))  #batch x hidden_size
        if train:
            agg = self.dropout(agg)
        return torch.sigmoid(self.linear2(agg))  
    
    def init_hidden(self, batch_size):
        return cuda_available(torch.zeros(2, batch_size, self.hidden_size))
        
#use glove
model_gru = GRU_Classifier(words_size, embedding_matrix, embedding_size=embedding_size, hidden_size=hidden_size,
                           max_len=max_len, dropout=dropout).cuda()
criterion = nn.BCELoss() 
optimizer = torch.optim.Adam(model_gru.parameters(), lr=lr)  #non-static
#optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  #static

#train process
train_loss_array = []  #keep total loss
starttime = datetime.datetime.now()
print("Training for %d epochs..." % epoch)
for i in range(epoch):
    train_loss = 0
    for j, (question, label) in enumerate(train_loader):
        optimizer.zero_grad()
        question = cuda_available(question).long()
        label = cuda_available(label)
        output = model_gru(question, train=True).squeeze(1)
        loss = criterion(output, label.float())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_loss_array.append(train_loss)
    endtime = datetime.datetime.now()
    print("epoch is %d, train_loss is %.4f, batch is %d, cost time is about %s" % 
          (i, train_loss, batch_size, changeTime((endtime - starttime).seconds)))
print("train finish!")

#build val_loader
#tensor_val_X = torch.from_numpy(val_X)
#tensor_val_y = torch.from_numpy(val_y)

#quoraValSet = TensorDataset(tensor_val_X, tensor_val_y)
#val_loader = DataLoader(dataset=quoraValSet,
#                         batch_size=batch_size,
#                         shuffle=False,
#                         num_workers=2)

#get val result
#val_cnn = []
#val_gru = []
#for question, lable in val_loader:
#    output_cnn = model_cnn(cuda_available(question).long(), train=False).squeeze(1)
#    output_gru = model_gru(cuda_available(question).long(), train=False).squeeze(1)
#    for i in range(len(question)):
#        val_cnn.append(output_cnn[i].item()) 
#        val_gru.append(output_gru[i].item()) 

#build test loader
tensor_test_X = torch.from_numpy(test_X)
quoraTestSet = TensorDataset(tensor_test_X)

test_loader = DataLoader(dataset=quoraTestSet,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=2)
                         
#get test result
test_cnn = []
test_gru = []
for question in test_loader:
    output_cnn = model_cnn(cuda_available(question[0]).long(), train=False).squeeze(1)
    output_gru = model_gru(cuda_available(question[0]).long(), train=False).squeeze(1)
    for i in range(len(question[0])):
        test_cnn.append(output_cnn[i].item()) 
        test_gru.append(output_gru[i].item()) 
        
#test on val_loader
#best_F1 = 0
#best_threshold = 0
#best_weight = 0.6
#for thresh in np.arange(0.1, 0.651, 0.01):
#    thresh = np.round(thresh, 2)
#    for i in np.arange(0.4, 0.901, 0.01):
#        TP, TN, FP, FN = 0, 0, 0, 0
#        weight = np.round(i, 2)
#        for j in range(len(val_cnn)):
#            output = weight * val_cnn[j] + (1 - weight) * val_gru[j] 
#            label = val_y[j]
#            if output > thresh:
#                predict1 = 1
#            else:
#                predict1 = 0
#            if predict1 == 1 and label == 1:
#                TP += 1
#            elif predict1 == 0 and label == 0:
#                TN += 1
#            elif predict1 == 0 and label == 1:
#                FN += 1
#            else:
#                FP += 1
#        p = TP / (TP + FP)
#        r = TP / (TP + FN)
#        F1 = 2 * r * p / (r + p)
#        acc = (TP + TN) / (TP + TN + FP + FN)
#        if F1 > best_F1:
#            best_F1 = F1
#            best_threshold = thresh
#            best_weight = weight
#        print("threshold {0}, weight {1}:F1 score={2}, Acc={3}".format(thresh, weight, F1, acc))
#print('----------------------------------------------------------------------')
#print('the best_F1 score={0} at threshold {1}, weight={2}'.format(best_F1, best_threshold, best_weight))

#predict
best_threshold = 0.33
best_weight = 0.6
predict = []
for i in range(len(test_cnn)):
    output = best_weight * test_cnn[i] + (1 - best_weight) * test_gru[i]
    if output > best_threshold:
        predict.append(1)
    else:
        predict.append(0)
        
submission['prediction'] = predict
submission.to_csv('submission.csv', index=False)
