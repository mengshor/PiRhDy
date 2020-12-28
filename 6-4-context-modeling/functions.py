import os
import pickle

from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import BatchNormalization, Concatenate
from tensorflow.keras.layers import Input, Dense, Bidirectional, GRU, Lambda, \
    Embedding, Add, Multiply, Masking
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import pandas as pd
import numpy as np

# useful layers
from layers import Attention
from keras_pos_embd import TrigPosEmbedding

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import ndcg_score
from math import sqrt



def get_embeddings():
    embedding_name = 'overall_con'
    ratio = 100
    np.random.seed(0)
    chroma_embeddings = np.load('embeddings/chroma_{}_{}.npy'.format(embedding_name, ratio))
    empty_weights = np.random.uniform(0.0, 1.0, (1, 1, 256))
    chroma_embeddings = np.concatenate((chroma_embeddings, empty_weights), axis=1)
    velocity_embeddings = np.load('embeddings/velocity_{}_{}.npy'.format(embedding_name, ratio))
    empty_weights = np.random.uniform(0.0, 1.0, (1, 1, 256))
    velocity_embeddings = np.concatenate((velocity_embeddings, empty_weights), axis=1)
    state_embeddings = np.load('embeddings/states_{}_{}.npy'.format(embedding_name, ratio))
    empty_weights = np.random.uniform(0.0, 1.0, (1, 1, 256))
    state_embeddings = np.concatenate((state_embeddings, empty_weights), axis=1)

    return chroma_embeddings, velocity_embeddings, state_embeddings


class Encoder:
    max_period = 8
    max_phrase = 2
    max_track = 11
    max_note = 128
    hidden_dim = 256
    act_func = 'tanh'
    layer_num = 2
    class_num = 1
    trainable_state = True
    phrase_model = None
    name = None
    is_hidden = False

    def __init__(self, phrase_model):
        self.phrase_model = phrase_model
        if self.phrase_model is not None:
            self.trainable_state = False
            self.is_hidden = True
        else:
            self.trainable_state = True

    def print_settings(self):
        print('=========================model configuration======================')
        print('name\t{}'.format(self.name))
        print('max_period\t{}'.format(self.max_period))
        print('max_phrase\t{}'.format(self.max_phrase))
        print('max_track\t{}'.format(self.max_track))
        print('max_note\t{}'.format(self.max_note))
        print('hidden_dimension\t{}'.format(self.hidden_dim))
        print('act_func\t{}'.format(self.act_func))
        print('layer_number\t{}'.format(self.layer_num))
        print('class_number\t{}'.format(self.class_num))
        print('trainable_state\t{}'.format(self.trainable_state))
        print('phrase_model\t{}'.format(self.phrase_model))
        print('is_hidden\t{}'.format(self.is_hidden))

    def note_encoder(self):
        input_data = Input((4 * self.max_note,),
                           name='input_data')
        input_chroma = Lambda(lambda x: x[:, :self.max_note],
                              name='input_chroma')(input_data)
        input_octave = Lambda(lambda x: x[:, self.max_note:2 * self.max_note],
                              name='input_octave')(input_data)
        input_velocity = Lambda(lambda x: x[:, 2 * self.max_note:3 * self.max_note],
                                name='input_velocity')(input_data)
        input_state = Lambda(lambda x: x[:, 3 * self.max_note:],
                             name='input_state')(input_data)

        # load embeddings
        chroma_embeddings, velocity_embeddings, state_embeddings = get_embeddings()
        chroma_embedding_layer = Embedding(input_dim=657,
                                           output_dim=256,
                                           mask_zero=True,
                                           weights=chroma_embeddings,
                                           trainable=self.trainable_state,
                                           name='chroma_embedding')
        velocity_embedding_layer = Embedding(input_dim=12,
                                             output_dim=256,
                                             mask_zero=True,
                                             weights=velocity_embeddings,
                                             trainable=self.trainable_state,
                                             name='velocity_embedding')
        state_embedding_layer = Embedding(input_dim=5,
                                          output_dim=256,
                                          mask_zero=True,
                                          weights=state_embeddings,
                                          trainable=self.trainable_state,
                                          name='state_embedding')

        # get embeddings
        chroma_rep = chroma_embedding_layer(input_chroma)
        velocity_rep = velocity_embedding_layer(input_velocity)
        state_rep = state_embedding_layer(input_state)

        # get octave representation
        octave_vector = Lambda(lambda x: K.mean(x, axis=1),
                               name='octave_embedding')(
            K.cast_to_floatx(chroma_embeddings))
        reshape_octave = Lambda(lambda x: K.expand_dims(x),
                                name='reshape_octave')

        # get pitch representation, namely fusing chroma and octave
        octave_reshape = reshape_octave(input_octave)
        octave_rep = Lambda(lambda x: x * octave_vector,
                            name='octave_rep')(octave_reshape)
        pitch_rep = Add(name='pitch_rep')([chroma_rep, octave_rep])

        # masking layers
        pitch_rep = Masking(mask_value=0.0)(pitch_rep)
        velocity_rep = Masking(mask_value=0.0)(velocity_rep)
        state_rep = Masking(mask_value=0.0)(state_rep)

        # batch_normalization layers
        pitch_rep = BatchNormalization()(pitch_rep)
        velocity_rep = BatchNormalization()(velocity_rep)
        state_rep = BatchNormalization()(state_rep)

        # generate position embeddings
        pitch_rep = TrigPosEmbedding(name="position_rep",
                                     output_dim=self.hidden_dim,
                                     mode='concat')(pitch_rep)

        note_rep = Concatenate(name='con_note')([pitch_rep,
                                                 velocity_rep,
                                                 state_rep])

        for i in range(self.layer_num):
            note_rep = TimeDistributed(Dense(self.hidden_dim,
                                             activation=self.act_func,
                                             trainable=self.trainable_state))(note_rep)

        note_encoder_model = Model(input_data,
                                   note_rep,
                                   name='note_encoder')
        # note_encoder_model.summary()
        return note_encoder_model

    def phrase_encoder(self):
        input_data = Input((4 * self.max_note,), name='input_phrase')
        note_rep = self.note_encoder()(input_data)
        note_rep = Masking(mask_value=0.0)(note_rep)
        note_rep = BatchNormalization()(note_rep)
        bi_gru_note = Bidirectional(GRU(self.hidden_dim // 2,
                                        activation=self.act_func,
                                        return_sequences=True,
                                        ),
                                    trainable=self.trainable_state,
                                    name='bi_gru_note')
        for i in range(self.layer_num):
            note_rep = bi_gru_note(note_rep)

        phrase_rep = Attention(attention_size=self.hidden_dim,
                               name='phrase_rep',
                               trainable=self.trainable_state)(note_rep)

        phrase_encoder_model = Model(input_data,
                                     phrase_rep,
                                     name='phrase_encoder')
        if not self.trainable_state:
            phrase_encoder_model.set_weights(self.get_weights())
        # phrase_encoder_model.summary()
        return phrase_encoder_model

    def get_weights(self):
        model_dict = {'next': next_phrase,
                      'acc': acc_ass}
        model_path = 'token_model/{}.h5'.format(self.phrase_model)
        model = model_dict[self.phrase_model]().encoder()
        model.load_weights(model_path)
        model_phrase = model.get_layer(index=1).get_layer(index=1).layer
        phrase_weights = model_phrase.get_weights()
        return phrase_weights

    def track_encoder(self):
        if not self.is_hidden:
            input_data = Input((self.max_track,
                                4 * self.max_note,),
                               name='input_track')
            phrase_rep = TimeDistributed(self.phrase_encoder())(input_data)
        else:
            input_data = Input((self.max_track,
                                self.hidden_dim,),
                               name='input_track')
            phrase_rep = input_data

        phrase_rep = Masking(mask_value=0.0)(phrase_rep)
        phrase_rep = BatchNormalization()(phrase_rep)
        track_rep = Attention(attention_size=self.hidden_dim,
                              name='track_rep')(phrase_rep)
        track_encoder_model = Model(input_data,
                                    track_rep,
                                    name='track_encoder')
        # track_encoder_model.summary()
        return track_encoder_model


    def melody_track_encoder(self):
        if not self.is_hidden:
            input_data = Input((self.max_phrase,
                                4 * self.max_note,),
                               name='input_melody')
            phrase_rep = TimeDistributed(self.phrase_encoder())(input_data)
        else:
            input_data = Input((self.max_phrase,
                                self.hidden_dim,),
                               name='input_melody')
            phrase_rep = input_data
        phrase_rep = Masking(mask_value=0.0)(phrase_rep)
        phrase_rep = BatchNormalization()(phrase_rep)
        bi_gru_phrase = Bidirectional(GRU(self.hidden_dim // 2,
                                          activation=self.act_func,
                                          return_sequences=True),
                                      name='bi_gru_phrase')
        for i in range(self.layer_num):
            phrase_rep = bi_gru_phrase(phrase_rep)
        melody_rep = Attention(attention_size=self.hidden_dim,
                               name='melody_rep')(phrase_rep)
        melody_encoder_model = Model(input_data,
                                     melody_rep,
                                     name='melody_track_encoder')
        # melody_encoder_model.summary()
        return melody_encoder_model


    def period_encoder(self):
        if not self.is_hidden:
            input_data = Input((self.max_phrase,
                                self.max_track,
                                4 * self.max_note,),
                               name='input_data')
        else:
            input_data = Input((self.max_phrase,
                                self.max_track,
                                self.hidden_dim,),
                               name='input_data')
        phrase_rep = TimeDistributed(self.track_encoder())(input_data)
        phrase_rep = Masking(mask_value=0.0)(phrase_rep)
        phrase_rep = BatchNormalization()(phrase_rep)
        bi_gru_phrase = Bidirectional(GRU(self.hidden_dim // 2, activation=self.act_func, return_sequences=True),
                                      name='bi_gru_phrase')
        for i in range(self.layer_num):
            phrase_rep = bi_gru_phrase(phrase_rep)
        period_rep = Attention(attention_size=self.hidden_dim, name='period_rep')(phrase_rep)
        period_encoder_model = Model(input_data,
                                     period_rep,
                                     name='period_encoder')
        # period_encoder_model.summary()
        return period_encoder_model


    def song_encoder(self, name):
        if not self.is_hidden:
            input_data = Input((self.max_period,
                                self.max_phrase,
                                self.max_track,
                                4 * self.max_note,), name='input_song')
        else:
            input_data = Input((self.max_period,
                                self.max_phrase,
                                self.max_track,
                                self.hidden_dim,), name='input_song')

        period_rep = TimeDistributed(self.period_encoder())(input_data)
        period_rep = Masking(mask_value=0.0)(period_rep)
        period_rep = BatchNormalization()(period_rep)
        bi_gru_period = Bidirectional(GRU(self.hidden_dim // 2,
                                          activation=self.act_func,
                                          return_sequences=True),
                                      name='bi_gru_period')
        for i in range(self.layer_num):
            period_rep = bi_gru_period(period_rep)
        song_rep = Attention(attention_size=self.hidden_dim, name='song_rep')(period_rep)
        song_encoder_model = Model(input_data,
                                   song_rep,
                                   name=name
                                   )
        # song_encoder_model.summary()
        return song_encoder_model


    def classifier(self):
        input_data = Input((self.hidden_dim,), name='input_rep')
        hidden_rep = Dense(self.hidden_dim // 2,
                           activation=self.act_func,
                           use_bias=True)(input_data)
        hidden_rep = BatchNormalization()(hidden_rep)
        labels = Dense(self.class_num, activation='sigmoid')(hidden_rep)
        classifier_model = Model(input_data,
                                 labels,
                                 name='classifier')
        # classifier_model.summary()
        return classifier_model


class next_phrase(Encoder):
    def __init__(self, phrase_model=None):
        super(next_phrase, self).__init__(phrase_model)
        self.max_period = None
        self.max_track = None

    def encoder(self):
        if not self.is_hidden:
            input_data = Input((self.max_phrase,
                                4 * self.max_note,),
                               name='input_next')
        else:
            input_data = Input((self.max_phrase,
                                self.hidden_dim,),
                               name='input_next')
        melody_rep = self.melody_track_encoder()(input_data)
        output = self.classifier()(melody_rep)
        return Model(input_data, output)


class acc_ass(Encoder):
    def __init__(self, phrase_model=None):
        super(acc_ass, self).__init__(phrase_model)
        self.max_period = None
        self.max_track = 2

    def encoder(self):
        if not self.is_hidden:
            input_data = Input((self.max_phrase,
                                4 * self.max_note,),
                               name='input_acc')
        else:
            input_data = Input((self.max_phrase,
                                self.hidden_dim,),
                               name='input_acc')
        melody_rep = self.track_encoder()(input_data)
        output = self.classifier()(melody_rep)
        return Model(input_data, output)


class genre_class(Encoder):
    def __init__(self, class_num, phrase_model):
        super(genre_class, self).__init__(phrase_model)
        self.class_num = class_num

    def encoder(self):
        if not self.is_hidden:
            input_data = Input((self.max_period,
                                self.max_phrase,
                                self.max_track,
                                4 * self.max_note,),
                               name='input_genre')
        else:
            input_data = Input((self.max_period,
                                self.max_phrase,
                                self.max_track,
                                self.hidden_dim,),
                               name='input_genre')
        song_rep = self.song_encoder(name='genre_song_encoder')(input_data)
        output = self.classifier()(song_rep)
        return Model(input_data, output)


class sim_rel(Encoder):
    def __init__(self, phrase_model):
        super(sim_rel, self).__init__(phrase_model)

    def encoder(self):

        if not self.is_hidden:
            input_data_0 = Input((self.max_period,
                                  self.max_phrase,
                                  self.max_track,
                                  4 * self.max_note,),
                                 name='input_song_0')
            input_data_1 = Input((self.max_period,
                                  self.max_phrase,
                                  self.max_track,
                                  4 * self.max_note,),
                                 name='input_song_1')
        else:
            input_data_0 = Input((self.max_period,
                                  self.max_phrase,
                                  self.max_track,
                                  self.hidden_dim,),
                                 name='input_song_0')
            input_data_1 = Input((self.max_period,
                                  self.max_phrase,
                                  self.max_track,
                                  self.hidden_dim,),
                                 name='input_song_1')
        song_rep_0 = self.song_encoder(name='song_encoder_0')(input_data_0)
        song_rep_1 = self.song_encoder(name='song_encoder_1')(input_data_1)
        dis_rep = Multiply(name='distance_rep')([song_rep_0, song_rep_1])
        output = self.classifier()(dis_rep)
        return Model([input_data_0, input_data_1], output)


class song_to_phrase(Encoder):
    def __init__(self, phrase_model):
        super(song_to_phrase, self).__init__(phrase_model)

    def encoder(self):
        input_data = Input((self.max_period,
                            self.max_phrase,
                            self.max_track,
                            4 * self.max_note,),
                           name='input_data')
        features = TimeDistributed(TimeDistributed(TimeDistributed(self.phrase_encoder())))(input_data)

        return Model(input_data, features, name='song_to_phrase')


class phrase_to_feature(Encoder):
    def __init__(self, phrase_model):
        super(phrase_to_feature, self).__init__(phrase_model)

    def encoder(self):
        input_data = Input((self.max_phrase,
                            4 * self.max_note,),
                           name='input_data')
        features = TimeDistributed(self.phrase_encoder())(input_data)

        return Model(input_data, features, name='phrase_to_feature')


class Data_Generator:
    batch_size = 1000
    data_path = ' '

    def __init__(self,
                 batch_size,
                 data_path,
                 phrase_model=None
                 ):
        self.batch_size = batch_size
        self.data_path = data_path
        self.phrase_model = phrase_model


class phrase_generator(Data_Generator):
    def __init__(self, batch_size, data_path, phrase_model=None):
        super().__init__(batch_size, data_path, phrase_model)
        if self.phrase_model is not None:
            self.input_dim = 512
        else:
            self.input_dim = 1024

    def train(self):
        while True:
            file = open(self.data_path + 'train')
            data = pd.read_csv(file,
                               decimal=',',
                               header=None,
                               dtype='a',
                               low_memory=False,
                               chunksize=self.batch_size)
            for idc, chunk in enumerate(data):
                feature = chunk.iloc[:, :self.input_dim].values
                feature = np.reshape(feature, (-1, 2, self.input_dim // 2))
                yield feature, chunk.iloc[:, -1]

            file.close()

    def train_rewrite(self, data_num):
        while True:
            file = open(self.data_path + 'train', 'rb')
            steps = np.ceil(data_num / self.batch_size)
            print(steps)
            for idx in range(int(steps)):
                print(idx)
                feature = np.loadtxt(file, delimiter=',',
                                     skiprows=self.batch_size * idx,
                                     max_rows=min(self.batch_size, (data_num - self.batch_size * idx)))
                feature = np.reshape(feature[:, :-1], (-1, 2, self.input_dim // 2))
                yield feature

            file.close()

    def test(self):
        while True:
            file = open(self.data_path + 'test_feature', 'rb')
            data = pd.read_csv(file,
                               decimal=',',
                               header=None,
                               dtype='a',
                               low_memory=False,
                               chunksize=self.batch_size)

            for chunk in data:
                feature = chunk.values
                feature = np.reshape(feature, (-1, 2, self.input_dim // 2))
                yield feature
            file.close()

    def test_rewrite(self, data_num):
        while True:
            file = open(self.data_path + 'test_feature', 'rb')
            steps = np.ceil(data_num / self.batch_size)
            print(steps)
            for idx in range(int(steps)):
                print(idx)
                feature = np.loadtxt(file, delimiter=',',
                                     skiprows=self.batch_size * idx,
                                     max_rows=min(self.batch_size, (data_num - self.batch_size * idx)))
                feature = np.reshape(feature, (-1, 2, self.input_dim // 2))
                yield feature
            file.close()




def get_ap(max_index, y_ranked):
    precision_score = 0.0
    hits = 0.0
    for idr, rank in enumerate(y_ranked):
        if rank < max_index:
            hits += 1.0
            precision_score += hits / (idr + 1.0)
    precision_score = precision_score / max_index

    return precision_score


def get_hits(max_index, y_ranked, k):
    hits = 0.0
    y_ranked = y_ranked[:k]
    for rank in y_ranked:
        if rank < max_index:
            hits += 1.0
    hits_at_k = hits / max_index

    return hits_at_k


class Metrics:
    y_true = None
    y_pred = None

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred


class rank(Metrics):
    def compute(self, placeholder=None):
        self.y_pred = np.squeeze(self.y_pred)
        data_num = len(self.y_true) // 50
        top_list = [1, 5, 10, 25]
        for top in top_list:
            locals()['hits' + str(top)] = 0.0
        precision = 0.0
        for sample_id in range(data_num):
            labels = self.y_true[50 * sample_id: 50 * (sample_id + 1)]
            max_index = 0.0
            for label in labels:
                if int(label) > 0:
                    max_index += 1.0
                else:
                    break
            scores = self.y_pred[50 * sample_id: 50 * (sample_id + 1)]
            ranked = np.argsort(-scores)
            precision += get_ap(max_index, ranked)
            for top in top_list:
                locals()['hits' + str(top)] += get_hits(max_index, ranked, top)
        precision = precision / data_num
        print('MAP = ' + str(precision))
        for top in top_list:
            locals()['hits' + str(top)] = locals()['hits' + str(top)] / data_num
            print('hits@{} = {}'.format(top, locals()['hits' + str(top)]))



def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_value = true_positives / (possible_positives + K.epsilon())
    return recall_value


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_value = true_positives / (predicted_positives + K.epsilon())
    return precision_value


def f1(y_true, y_pred):
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    return 2 * ((precision_value * recall_value) / (precision_value + recall_value + K.epsilon()))


def rewrite_song(phrase_model):
    dataset_path = 'dataset/song_dataset/'
    feature_path = 'features/song_dataset_{}/'.format(phrase_model)
    model = song_to_phrase(phrase_model=phrase_model).encoder()
    for root, dirs, files in os.walk(dataset_path):
        for dir_name in dirs:
            if not os.path.exists(os.path.join(feature_path, dir_name)):
                os.makedirs(os.path.join(feature_path, dir_name))
        for file in files:
            file_path = os.path.join(root, file)
            data = np.load(file_path)
            features = model.predict(np.expand_dims(data, axis=0))
            features = np.squeeze(features)
            np.save(feature_path + file[0] + '/' + file, features)
    print('finish')


def rewrite_phrase(phrase_name, is_test=False):
    config_list = {'next': ['acc', 15781108, 10015000],
                   'acc': ['next', 3569888, 9963500]}
    task_name = config_list[phrase_name][0]
    train_num = config_list[phrase_name][1]
    test_num = config_list[phrase_name][2]
    batch_size = 2000
    dataset_path = 'dataset/{}/'.format(task_name)
    model = phrase_to_feature(phrase_model=phrase_name).encoder()
    # model.summary()
    if not is_test:
        data = pd.read_csv(open(dataset_path + 'train'),
                           decimal=',',
                           header=None,
                           dtype='a',
                           low_memory=False,
                           chunksize=batch_size * 100)
        file_name = open('features/{}/train_features'.format(task_name), 'ab')
        # steps = np.ceil(train_num / batch_size)
        # generator = phrase_generator(batch_size, dataset_path).train_rewrite(train_num)
        for chunk in data:
            feature = chunk.iloc[:, :1024].values
            feature = np.reshape(feature, (-1, 2, 512))
            features = model.predict(feature, verbose=1, batch_size=batch_size)
            features = features.reshape((-1, 512))
            np.savetxt(file_name, features, delimiter=',')
    else:
        data = pd.read_csv(open(dataset_path + 'test_feature'),
                           decimal=',',
                           header=None,
                           dtype='a',
                           low_memory=False,
                           chunksize=batch_size * 100)
        file_name = open('features/{}/test_features_16'.format(task_name), 'ab')
        # steps = np.ceil(test_num / batch_size)
        # generator = phrase_generator(batch_size, dataset_path).test_rewrite(test_num)
        for chunk in data:
            feature = chunk.iloc[:, :1024].values
            feature = np.reshape(feature, (-1, 2, 512))
            features = model.predict(feature, verbose=1, batch_size=batch_size)
            features = features.reshape((-1, 512))
            np.savetxt(file_name, features, delimiter=',')

    # features = model.predict(generator, verbose=1, steps=steps)
    # features = features.reshape((-1, 512))
    # np.savetxt(file_name, features, delimiter=',')
    print('finish')




def phrase_to_melody(name):
    data = pd.read_csv(open('dataset/next/{}'.format(name), 'r'), header=None, index_col=None, chunksize=200000,
                       low_memory=False)
    new_file = open('dataset/next/{}_melody'.format(name), 'w')
    for chunk in data:
        feature = chunk.iloc[:, :1024].values
        feature = np.reshape(feature, (-1, 2, 512))
        for value in feature:
            phrase_str = write_phrase_to_str(value[0]) + ',' + \
                         write_phrase_to_str(value[1]) + '\n'
            new_file.write(phrase_str)
    new_file.close()
    print('finish')


def write_phrase_to_str(phrase):
    phrase_str = ''
    chroma = list(phrase[:128])
    octave = list(phrase[128:256])
    state = phrase[384:512]
    indexes = np.squeeze(np.argwhere(state == 3))
    if indexes[0] != 0:
        indexes = np.insert(indexes, 0, 0)
    pitch = [chroma[index] - 1 +
             12 * (octave[index] - 1) for index in indexes]
    durations = []

    durations.extend([indexes[i + 1] - item for i, item in enumerate(indexes[:-1])])
    for i, state_value in enumerate(state[indexes[-1] + 1:]):
        if state_value == 0:
            durations.extend([i + 1])
            break

    for i in range(len(pitch)):
        phrase_str = phrase_str + str(pitch[i]) + '-' + str(durations[i]) + '*'
    return phrase_str


def song_to_melody():
    dataset_path = 'dataset/song_dataset/'
    new_path = 'dataset/song_dataset_melody/'
    for root, dirs, files in os.walk(dataset_path):
        for dir_name in dirs:
            if not os.path.exists(os.path.join(new_path, dir_name)):
                os.makedirs(os.path.join(new_path, dir_name))
        for file in files:
            file_path = os.path.join(root, file)
            data = np.load(file_path)
            data = np.squeeze(data[:, :, 0, :])
            print(data.shape)
            file_name = new_path + file[0] + '/' + file
            new_file = open(file_name, 'w')
            for period in data:
                new_file.write(write_phrase_to_str(period[0]) + ',' +
                               write_phrase_to_str(period[1]) + '\n')
            new_file.close()
    print('finish')
