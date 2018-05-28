
# coding: utf-8

# In[9]:

import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
from sklearn.model_selection import train_test_split
import itertools
from keras.preprocessing.sequence import pad_sequences



# In[3]:

TRAIN_CSV = 'final_train.csv'
TEST_CSV= 'sts-test - sts-test.csv'

EMBEDDING_FILE = '/home/vinisha/Documents/semproject/GoogleNews-vectors-negative300.bin.gz'
MODEL_SAVING_DIR = '/home/vinisha/Documents/semproject/sem4/'


# In[5]:

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV,delimiter='\t')

# stops = set(stopwords.words('english'))

def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
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

    text = text.split()

    return text

# Prepare embedding
vocabulary = dict()
inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

questions_cols = ['S1', 'S2']

# Iterate over the questions only of both training and test datasets
for dataset in [train_df, test_df]:
    for index, row in dataset.iterrows():

        # Iterate through the text of both questions of the row
        for question in questions_cols:

            q2n = []  # q2n -> question numbers representation
            for word in text_to_word_list(row[question]):

                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    q2n.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    q2n.append(vocabulary[word])

            # Replace questions as word to question as number representation
            dataset.set_value(index, question, q2n)
            
embedding_dim = 300
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
embeddings[0] = 0  # So that the padding will be ignored

# Build the embedding matrix
for word, index in vocabulary.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)


# In[13]:

max_seq_length = max(train_df.S1.map(lambda x: len(x)).max(),
                     train_df.S2.map(lambda x: len(x)).max(),
                     test_df.S1.map(lambda x: len(x)).max(),
                     test_df.S2.map(lambda x: len(x)).max())

# Split to train validation
validation_size = 1000
training_size = len(train_df) - validation_size

X = train_df[questions_cols]
Y = (train_df['SCORE']-1)/4.0

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

# Split to dicts
X_train = {'left': X_train.S1, 'right': X_train.S2}
X_validation = {'left': X_validation.S1, 'right': X_validation.S2}
X_test = {'left': test_df.S1, 'right': test_df.S2}

# Convert labels to their numpy representations
Y_train1 = (Y_train.values-1)/4.00
Y_validation1 = (Y_validation.values-1)/4.00
# Y_train1 = (Y_train.values)
# Y_validation1 = (Y_validation.values)

# Zero padding
for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)


# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)


# In[16]:

import pickle
with open("vocabulary" + '.pkl', 'wb') as f:
    pickle.dump(vocabulary, f, pickle.HIGHEST_PROTOCOL)
with open("embedding" + '.pkl', 'wb') as f:
    pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)
with open("embedding_dim" + '.pkl', 'wb') as f:
    pickle.dump(embedding_dim, f, pickle.HIGHEST_PROTOCOL)

with open("inverse_vocabulary" + '.pkl', 'wb') as f:
    pickle.dump(inverse_vocabulary, f, pickle.HIGHEST_PROTOCOL)
with open("max_seq_length" + '.pkl', 'wb') as f:
    pickle.dump(max_seq_length, f, pickle.HIGHEST_PROTOCOL)
    
with open("X_Train" + '.pkl', 'wb') as f:
    pickle.dump(X_train, f, pickle.HIGHEST_PROTOCOL)
    
with open("X_Validation" + '.pkl', 'wb') as f:
    pickle.dump(X_validation, f, pickle.HIGHEST_PROTOCOL)
    
with open("X_Test" + '.pkl', 'wb') as f:
    pickle.dump(X_test, f, pickle.HIGHEST_PROTOCOL)
    
with open("Y_Train" + '.pkl', 'wb') as f:
    pickle.dump(Y_train1, f, pickle.HIGHEST_PROTOCOL)
    
with open("Y_Validation" + '.pkl', 'wb') as f:
    pickle.dump(Y_validation1, f, pickle.HIGHEST_PROTOCOL)

