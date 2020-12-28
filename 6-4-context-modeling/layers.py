from tensorflow.keras.layers import Embedding, Layer
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import Callback
import time


class Attention(Layer):
    def __init__(self, attention_size, **kwargs):
        self.attention_size = attention_size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, ATTENTION_SIZE)
        # b: (ATTENTION_SIZE, 1)
        # u: (ATTENTION_SIZE, 1)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(self.attention_size, 1),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS, ATTENTION_SIZE)
        et = K.tanh(K.dot(x, self.W) + self.b)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        output = K.sum(ot, axis=1)
        return output

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class EmbeddingRet(Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return [super(EmbeddingRet, self).compute_output_shape(input_shape),
                (self.input_dim, self.output_dim),
                # (self.output_dim, 1)
                ]

    def compute_mask(self, inputs, mask=None):
        return [super(EmbeddingRet, self).compute_mask(inputs, mask),
                None,
                ]

    def call(self, inputs):
        return [super(EmbeddingRet, self).call(inputs),
                self.embeddings
                # K.expand_dims(K.mean(self.embeddings, axis=0), axis=0),
                ]


class Position_Embedding(Layer):

    def __init__(self, embedding_dim=256, **kwargs):
        self.embedding_dim = embedding_dim
        super(Position_Embedding, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.embedding_dim)

    def call(self, positions):
        batch_size, sequence_length = K.shape(positions)[0], K.int_shape(positions)[1]
        # 生成(self.embedding_dim,)向量：1/(10000^(2*[0,1,2,...,self.embedding_dim-1]/self.embedding_dim))，对应公式中的1/10000^(2i/d_pos)
        embedding_wise_pos = 1. / K.pow(10000., 2 * K.arange(self.embedding_dim / 2,
                                                             dtype='float32') / self.embedding_dim)  # n_dims=1, shape=(self.embedding_dim,)
        # 增加维度
        embedding_wise_pos = K.expand_dims(embedding_wise_pos, 0)  # n_dims=2, shape=(1,self.embedding_dim)
        # 增加维度
        word_wise_pos = K.expand_dims(positions, 2)  # n_dims=3, shape=(batch_size,sequence_length,1)
        # 生成(batch_size,sequence_length,self.embedding_dim)向量，对应公式中的p/10000^(2i/d_pos)
        position_embeddings = K.dot(word_wise_pos, embedding_wise_pos)
        # 直接concatenate无法出现交替现象，应先升维再concatenate再reshape

        position_embeddings = K.reshape(
            K.concatenate([K.cos(position_embeddings), K.sin(position_embeddings)], axis=-1),
            shape=(-1, sequence_length, self.embedding_dim))
        return position_embeddings


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, data_file, data_number, batch_size, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.data_number = data_number
        self.shuffle = shuffle
        self.on_epoch_end()
        self.data_file = data_file

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data_number / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        skip = int(self.indexes[index] * self.batch_size)
        x_data = pd.read_csv(self.data_file,
                             delimiter=',',
                             skiprows=skip,
                             nrows=self.batch_size,
                             header=None,
                             index_col=False,
                             dtype='a'
                             )
        print('finish load')
        x_bon = x_data.iloc[:, 0:25]
        x_harmony = x_data.iloc[:, 25:129]
        x_label = x_data.iloc[:, 129:]
        return [x_bon.values, x_harmony.values], x_label.values

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(np.ceil(self.data_number / self.batch_size))
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)
