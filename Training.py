
# coding: utf-8

# In[18]:

import pickle
from Model import getModel
from keras.optimizers import Adadelta
from time import time
import datetime


# In[11]:

def loadPickle(name):
    with open(name+ '.pkl', 'rb') as f:
        return pickle.load(f)


# In[14]:

X_train = loadPickle("X_Train")
X_validation = loadPickle("X_Validation")
Y_validation1 = loadPickle("Y_Validation")
Y_train1 = loadPickle("Y_Train")


# In[17]:

malstm = getModel()
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 16
n_epoch = 1
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

# Start training
training_start_time = time()

malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train1, batch_size=batch_size, nb_epoch=n_epoch,validation_data=([X_validation['left'], X_validation['right']], Y_validation1))

print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))


# In[20]:

malstm.save_weights('malstm.h5')

