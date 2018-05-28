
# coding: utf-8

# In[48]:

from keras.layers import dot
import tensorflow as tf
# from Attetntion import AttentionWithContext,dot_product,regularizers,constraints,initializers
from keras import initializers, regularizers, constraints
from keras.constraints import max_norm
import pickle
from keras.layers import Input, Embedding, LSTM, Merge,Concatenate,Dense,Bidirectional,Layer
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.models import Model



# In[45]:

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        print input_shape
        print "kkk"
        assert len(input_shape[1]) == 3
        self.seq_len=input_shape[1][1]
        self.W = self.add_weight((input_shape[1][-1], input_shape[1][-1],),
                                   name='W',
                                   initializer="random_uniform",
                                   regularizer=regularizers.l2(0.01),
                                   constraint=max_norm(2.))
        self.W_a = self.add_weight((input_shape[1][-1], input_shape[1][-1],),
                                 name='W_a',
                                   initializer="random_uniform",
                                   regularizer=regularizers.l2(0.01),
                                   constraint=max_norm(2.))
        
        if self.bias:
            self.b = self.add_weight((input_shape[1][-1],),
                                     name='b',
                                   initializer="random_uniform",
                                   regularizer=regularizers.l2(0.01),
                                   constraint=max_norm(2.))

        self.u = self.add_weight((input_shape[1][-1],),
                                 name='u',
                                   initializer="random_uniform",
                                   regularizer=regularizers.l2(0.01),
                                   constraint=max_norm(2.))

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x[1], self.W)

#         if self.bias:
        s=x[0]
        s1 = K.repeat(s, self.seq_len)
        _Wxstm = K.dot(s1, self.W_a)
        uit += _Wxstm

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)
        print a
        # apply mask after the exp. will be re-normalized next
#         if mask is not None:
#             # Cast the mask to floatX to avoid float64 upcasting in theano
#             a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x[1] * a
        print weighted_input.get_shape()
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[1][0], input_shape[1][-1]


# In[5]:

def loadPickle(name):
    with open(name+ '.pkl', 'rb') as f:
        return pickle.load(f)


# In[10]:

def getModel():
    embeddings = loadPickle("embedding")
    max_seq_length = loadPickle("max_seq_length")
    embedding_dim = loadPickle("embedding_dim")
    n_hidden = 50
    gradient_clipping_norm = 1.25
    batch_size = 16
    n_epoch = 80
    # def f1score(y_true, y_pred):
    #   return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

    def exponent_neg_manhattan_distance(left, right):
        ''' Helper function for the similarity estimate of the LSTMs outputs'''
        print (K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True)))
        return (K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True)))

    # The visible layer
    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')

    embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False,mask_zero=True if max_seq_length> 0 else False,)

    # Embedded version of the inputs
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)


    # In[46]:

    shared_lstm= LSTM(n_hidden,return_sequences=True,return_state=True)
    new_shared_lstm =Bidirectional(shared_lstm, merge_mode='concat', weights=None)

    left_output,h_sl,c_vl = shared_lstm(encoded_left)
    right_output,h_sr,c_vr = shared_lstm(encoded_right)

    # print left_output.get_shape()
    # c_vl = K.repeat(c_vl, max_seq_length)
    # c_vr = K.repeat(c_vr, max_seq_length)

    # right_ip=Concatenate([c_vl,right_output])
    # left_ip=Concatenate([c_vr,left_output])
    attention = AttentionWithContext()
    left_op= attention([c_vr,left_output])
    right_op=attention([c_vl,right_output])
    dense = Dense(25)
    d_left = dense(left_op)
    d_right= dense(right_op)


    # In[49]:

    malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([d_left,d_right])
    malstm = Model(inputs=[left_input, right_input],outputs=[malstm_distance])
    return malstm

if __name__ == '__main__':
    model =getModel()

