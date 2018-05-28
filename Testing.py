
# coding: utf-8

# In[9]:

import pickle
from Model import getModel
from keras.optimizers import Adadelta
from time import time
import itertools
import datetime
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np


# In[3]:

def loadPickle(name):
    with open(name+ '.pkl', 'rb') as f:
        return pickle.load(f)


# In[5]:

vocab = loadPickle("vocabulary")
inverse_vocabulary = loadPickle("inverse_vocabulary")
embeddings = loadPickle("embedding")
max_seq_length = loadPickle("max_seq_length")
embedding_dim = loadPickle("embedding_dim")
X_test = loadPickle("X_Test")
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 16

# test_sample = {'S1': S1,'S2':S2}
# test_sample_enocded =dict()
malstm = getModel()
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
malstm.load_weights("malstm.h5")


for dataset, side in itertools.product([X_test], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

assert X_test['left'].shape == X_test['right'].shape
# print X_train['left'].shape
# test=[X_test['left'][0],X_test['right'][0]]
malstm.summary()
# op=[]
op=((malstm.predict([X_test['left'], X_test['right']], batch_size=batch_size))*4)+1


# In[10]:

result = pd.DataFrame(op)
result1=result.round(1)
n1=np.array(result1)
np.savetxt('np111111.csv', n1, fmt='%.1f')


# In[11]:

# print n1

