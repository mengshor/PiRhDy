from tensorflow.keras.layers import Input, Embedding, BatchNormalization, Masking, GlobalAveragePooling1D, \
    Dense, GRU, Lambda, Average, Add, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from layers import Position_Embedding, EmbeddingRet
import tensorflow.keras.backend as K
import pandas as pd


def generate_batch_data(chunk_size, data_path):
    while True:
        file = open(data_path)
        data = pd.read_csv(file, header=None, decimal=',', chunksize=chunk_size, dtype='float32', low_memory=False)
        for chunk_id, chunk in enumerate(data):
            melody = chunk.iloc[:, 0:25]
            harmony = chunk.iloc[:, 25:129]
            label = chunk.iloc[:, 129:]
            yield [melody, harmony], label
        file.close()


def generate_melody_batch_data(chunk_size, data_path):
    while True:
        file = open(data_path)
        data = pd.read_csv(file, header=None, decimal=',', chunksize=chunk_size, dtype='float32', low_memory=False)
        for chunk_id, chunk in enumerate(data):
            melody = chunk.iloc[:, 0:25]
            label = chunk.iloc[:, 129:]
            yield melody, label
        file.close()


def generate_harmony_batch_data(chunk_size, data_path):
    while True:
        file = open(data_path)
        data = pd.read_csv(file, header=None, decimal=',', chunksize=chunk_size, dtype='float32', low_memory=False)
        for chunk_id, chunk in enumerate(data):
            harmony = chunk.iloc[:, 25:129]
            label = chunk.iloc[:, 129:]
            yield harmony, label
        file.close()


def get_info(x, index, info_num, total_num):
    temp = K.transpose(K.stack([x[:, index + idx * info_num] for idx in range(total_num // info_num)]))
    return temp


def token_modeling_con(act_func, hidden_dim, layer_number):
    # configuration
    chroma_num = 655
    velocity_num = 10
    state_num = 3

    # input layers
    input_melody = Input((25,), name='input_melody', dtype=float)
    input_harmony = Input((104,), name='input_harmony', dtype=float)

    # get info from melody, window=2
    melody_position = Lambda(name='melody_position', function=get_info, output_shape=(5,),
                          arguments={'index': 0, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_chroma = Lambda(name='melody_chroma', function=get_info, output_shape=(5,),
                        arguments={'index': 1, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_octave = Lambda(name='melody_octave', function=get_info, output_shape=(5,),
                        arguments={'index': 2, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_velocity = Lambda(name='melody_velocity', function=get_info, output_shape=(5,),
                          arguments={'index': 3, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_state = Lambda(name='melody_state', function=get_info, output_shape=(5,),
                       arguments={'index': 4, 'info_num': 5, 'total_num': 25})(input_melody)

    # get info from harmony, max length= 25+1 note events
    harmony_chroma = Lambda(get_info, name='harmony_chroma', output_shape=(26,),
                            arguments={'index': 0, 'info_num': 4, 'total_num': 104})(
        input_harmony)
    harmony_octave = Lambda(get_info, name='harmony_octave', output_shape=(26,),
                            arguments={'index': 1, 'info_num': 4, 'total_num': 104})(
        input_harmony)
    harmony_velocity = Lambda(get_info, name='harmony_velocity', output_shape=(26,),
                              arguments={'index': 2, 'info_num': 4, 'total_num': 104})(
        input_harmony)
    harmony_state = Lambda(get_info, name='harmony_state', output_shape=(26,),
                           arguments={'index': 3, 'info_num': 4, 'total_num': 104})(
        input_harmony)

    # embedding layers
    position_embedding = Position_Embedding(name='postion_embedding')
    chroma_embedding = EmbeddingRet(input_dim=chroma_num + 1, output_dim=hidden_dim, mask_zero=True,
                                    name='chroma_embedding')
    velocity_embedding = Embedding(input_dim=velocity_num + 1, output_dim=hidden_dim, mask_zero=True,
                                   name='velocity_embedding')
    state_embedding = Embedding(input_dim=state_num + 1, output_dim=hidden_dim, mask_zero=True, name='state_embedding')

    # get embeddings and mask zero
    melody_position_rep = position_embedding(melody_position)
    melody_chroma_rep, chroma_matirx = chroma_embedding(melody_chroma)
    melody_chroma_rep = Masking(mask_value=0.0)(melody_chroma_rep)
    melody_velocity_rep = velocity_embedding(melody_velocity)
    melody_velocity_rep = Masking(mask_value=0.0)(melody_velocity_rep)
    melody_state_rep = state_embedding(melody_state)
    melody_state_rep = Masking(mask_value=0.0)(melody_state_rep)

    harmony_chroma_rep, _ = chroma_embedding(harmony_chroma)
    harmony_chroma_rep = Masking(mask_value=0.0)(harmony_chroma_rep)
    harmony_velocity_rep = velocity_embedding(harmony_velocity)
    harmony_velocity_rep = Masking(mask_value=0.0)(harmony_velocity_rep)
    harmony_state_rep = state_embedding(harmony_state)
    harmony_state_rep = Masking(mask_value=0.0)(harmony_state_rep)

    # get octave representation
    octave_rep = Lambda(lambda x: K.mean(x, axis=0), name='octave_embedding')(chroma_matirx)
    reshape_octave = Lambda(lambda x: K.expand_dims(x), name='reshape_octave')

    # get pitch representation, namely fusing chroma and octave
    melody_octave_reshape = reshape_octave(melody_octave)
    melody_octave_rep = Lambda(lambda x: x * octave_rep, name='melody_octave_rep')(melody_octave_reshape)
    melody_octave_rep = Masking(mask_value=0.0)(melody_octave_rep)
    melody_pitch_rep = Add(name='melody_pitch_rep')([melody_chroma_rep, melody_octave_rep])
    melody_pitch_rep = BatchNormalization()(melody_pitch_rep)

    harmony_octave_reshape = reshape_octave(harmony_octave)
    harmony_octave_rep = Lambda(lambda x: x * octave_rep, name='harmony_octave_rep')(harmony_octave_reshape)
    harmony_octave_rep = Masking(mask_value=0.0)(harmony_octave_rep)
    harmony_pitch_rep = Add(name='harmony_pitch_rep')([harmony_chroma_rep, harmony_octave_rep])
    harmony_pitch_rep = Masking(mask_value=0.0)(harmony_pitch_rep)
    harmony_pitch_rep = BatchNormalization()(harmony_pitch_rep)
    # get note representation
    melody_note_rep = Concatenate(name='melody_note_representation')(
        [melody_position_rep, melody_pitch_rep, melody_velocity_rep, melody_state_rep])
    harmony_note_rep = Concatenate(name='harmony_note_representation')(
        [harmony_pitch_rep, harmony_velocity_rep, harmony_state_rep])

    for i in range(layer_number):
        melody_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(melody_note_rep)
        harmony_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(harmony_note_rep)

    harmony_rep = GlobalAveragePooling1D(name='harmony_rep')(harmony_note_rep)
    harmony_rep = BatchNormalization()(harmony_rep)
    for i in range(layer_number - 1):
        melody_note_rep = Bidirectional(GRU(hidden_dim // 2, activation=act_func, return_sequences=True))(melody_note_rep)
    melody_rep = Bidirectional(GRU(hidden_dim, activation=act_func), name='melody_rep')(melody_note_rep)
    melody_rep = BatchNormalization()(melody_rep)

    # get outputs
    melody_outputs = Dense(4, activation='sigmoid', name='melody_output')(melody_rep)
    harmony_output = Dense(4, activation='sigmoid', name='harmony_output')(harmony_rep)
    outputs = Concatenate(name='outputs')([melody_outputs, harmony_output])
    label_output = Dense(4, activation='sigmoid', name='label_output')(outputs)

    local_model_con = Model(inputs=[input_melody, input_harmony], outputs=label_output)
    local_model_con.summary()
    return local_model_con


def token_modeling(act_func, hidden_dim, layer_number):
    # configuration
    chroma_num = 655
    velocity_num = 10
    state_num = 3

    # input layers
    input_melody = Input((25,), name='input_melody', dtype=float)
    input_harmony = Input((104,), name='input_harmony', dtype=float)

    # get info from melody, window=2
    melody_position = Lambda(name='melody_position', function=get_info, output_shape=(5,),
                          arguments={'index': 0, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_chroma = Lambda(name='melody_chroma', function=get_info, output_shape=(5,),
                        arguments={'index': 1, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_octave = Lambda(name='melody_octave', function=get_info, output_shape=(5,),
                        arguments={'index': 2, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_velocity = Lambda(name='melody_velocity', function=get_info, output_shape=(5,),
                          arguments={'index': 3, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_state = Lambda(name='melody_state', function=get_info, output_shape=(5,),
                       arguments={'index': 4, 'info_num': 5, 'total_num': 25})(input_melody)

    # get info from harmony, max length= 25+1 note events
    harmony_chroma = Lambda(get_info, name='harmony_chroma', output_shape=(26,),
                            arguments={'index': 0, 'info_num': 4, 'total_num': 104})(
        input_harmony)
    harmony_octave = Lambda(get_info, name='harmony_octave', output_shape=(26,),
                            arguments={'index': 1, 'info_num': 4, 'total_num': 104})(
        input_harmony)
    harmony_velocity = Lambda(get_info, name='harmony_velocity', output_shape=(26,),
                              arguments={'index': 2, 'info_num': 4, 'total_num': 104})(
        input_harmony)
    harmony_state = Lambda(get_info, name='harmony_state', output_shape=(26,),
                           arguments={'index': 3, 'info_num': 4, 'total_num': 104})(
        input_harmony)

    # embedding layers
    position_embedding = Position_Embedding(name='postion_embedding')
    chroma_embedding = EmbeddingRet(input_dim=chroma_num + 1, output_dim=hidden_dim, mask_zero=True,
                                    name='chroma_embedding')
    velocity_embedding = Embedding(input_dim=velocity_num + 1, output_dim=hidden_dim, mask_zero=True,
                                   name='velocity_embedding')
    state_embedding = Embedding(input_dim=state_num + 1, output_dim=hidden_dim, mask_zero=True, name='state_embedding')

    # get embeddings and mask zero
    melody_position_rep = position_embedding(melody_position)
    melody_chroma_rep, chroma_matirx = chroma_embedding(melody_chroma)
    melody_chroma_rep = Masking(mask_value=0.0)(melody_chroma_rep)
    melody_velocity_rep = velocity_embedding(melody_velocity)
    melody_velocity_rep = Masking(mask_value=0.0)(melody_velocity_rep)
    melody_state_rep = state_embedding(melody_state)
    melody_state_rep = Masking(mask_value=0.0)(melody_state_rep)

    harmony_chroma_rep, _ = chroma_embedding(harmony_chroma)
    harmony_chroma_rep = Masking(mask_value=0.0)(harmony_chroma_rep)
    harmony_velocity_rep = velocity_embedding(harmony_velocity)
    harmony_velocity_rep = Masking(mask_value=0.0)(harmony_velocity_rep)
    harmony_state_rep = state_embedding(harmony_state)
    harmony_state_rep = Masking(mask_value=0.0)(harmony_state_rep)

    # get octave representation
    octave_rep = Lambda(lambda x: K.mean(x, axis=0), name='octave_embedding')(chroma_matirx)
    reshape_octave = Lambda(lambda x: K.expand_dims(x), name='reshape_octave')

    # get pitch representation, namely fusing chroma and octave
    melody_octave_reshape = reshape_octave(melody_octave)
    melody_octave_rep = Lambda(lambda x: x * octave_rep, name='melody_octave_rep')(melody_octave_reshape)
    melody_octave_rep = Masking(mask_value=0.0)(melody_octave_rep)
    melody_pitch_rep = Add(name='melody_pitch_rep')([melody_chroma_rep, melody_octave_rep])
    melody_pitch_rep = BatchNormalization()(melody_pitch_rep)

    harmony_octave_reshape = reshape_octave(harmony_octave)
    harmony_octave_rep = Lambda(lambda x: x * octave_rep, name='harmony_octave_rep')(harmony_octave_reshape)
    harmony_octave_rep = Masking(mask_value=0.0)(harmony_octave_rep)
    harmony_pitch_rep = Add(name='harmony_pitch_rep')([harmony_chroma_rep, harmony_octave_rep])
    harmony_pitch_rep = Masking(mask_value=0.0)(harmony_pitch_rep)
    harmony_pitch_rep = BatchNormalization()(harmony_pitch_rep)
    # get note representation
    melody_note_rep = Concatenate(name='melody_note_representation')(
        [melody_position_rep, melody_pitch_rep, melody_velocity_rep, melody_state_rep])
    harmony_note_rep = Concatenate(name='harmony_note_representation')(
        [harmony_pitch_rep, harmony_velocity_rep, harmony_state_rep])

    for i in range(layer_number):
        melody_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(melody_note_rep)
        harmony_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(harmony_note_rep)

    harmony_rep = GlobalAveragePooling1D(name='harmony_rep')(harmony_note_rep)
    harmony_rep = BatchNormalization()(harmony_rep)
    for i in range(layer_number - 1):
        melody_note_rep = Bidirectional(GRU(hidden_dim // 2, activation=act_func, return_sequences=True))(melody_note_rep)
    melody_rep = Bidirectional(GRU(hidden_dim, activation=act_func), name='melody_rep')(melody_note_rep)
    melody_rep = BatchNormalization()(melody_rep)

    # get outputs
    melody_outputs = Dense(4, activation='sigmoid', name='melody_output')(melody_rep)
    harmony_output = Dense(4, activation='sigmoid', name='harmony_output')(harmony_rep)
    label_output = Average(name='label_output')([melody_outputs, harmony_output])

    local_model = Model(inputs=[input_melody, input_harmony], outputs=label_output)
    local_model.summary()
    return local_model


def chroma_modeling(act_func, hidden_dim, layer_number):
    # analysis chroma info
    print('modeling chroma info')
    # configuration
    chroma_num = 655

    # input layers
    input_melody = Input((25,), name='input_melody')
    input_harmony = Input((104,), name='input_harmony')

    # get info from melody, window=2
    melody_chroma = Lambda(name='melody_chroma', function=get_info, output_shape=(5,),
                        arguments={'index': 1, 'info_num': 5, 'total_num': 25})(input_melody)
    # get info from harmony, max length= 25+1 note events
    harmony_chroma = Lambda(get_info, name='harmony_chroma', output_shape=(26,),
                            arguments={'index': 0, 'info_num': 4, 'total_num': 104})(
        input_harmony)

    # embedding layers
    chroma_embedding = EmbeddingRet(input_dim=chroma_num + 1, output_dim=hidden_dim, mask_zero=True,
                                    name='chroma_embedding')

    # get embeddings and mask zero
    melody_chroma_rep, _ = chroma_embedding(melody_chroma)
    melody_chroma_rep = Masking(mask_value=0.0)(melody_chroma_rep)

    harmony_chroma_rep, _ = chroma_embedding(harmony_chroma)
    harmony_chroma_rep = Masking(mask_value=0.0)(harmony_chroma_rep)

    # get note representation
    melody_note_rep = melody_chroma_rep
    harmony_note_rep = harmony_chroma_rep

    for i in range(layer_number):
        melody_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(melody_note_rep)
        harmony_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(harmony_note_rep)

    harmony_rep = GlobalAveragePooling1D(name='harmony_rep')(harmony_note_rep)
    harmony_rep = BatchNormalization()(harmony_rep)
    for i in range(layer_number - 1):
        melody_note_rep = Bidirectional(GRU(hidden_dim // 2, activation=act_func, return_sequences=True))(melody_note_rep)
    melody_rep = Bidirectional(GRU(hidden_dim, activation=act_func), name='melody_rep')(melody_note_rep)
    melody_rep = BatchNormalization()(melody_rep)
    # get outputs
    melody_outputs = Dense(4, activation='sigmoid', name='melody_output')(melody_rep)
    harmony_output = Dense(4, activation='sigmoid', name='harmony_output')(harmony_rep)
    label_output = Average(name='label_output')([melody_outputs, harmony_output])

    chroma_model = Model(inputs=[input_melody, input_harmony], outputs=label_output)
    chroma_model.summary()
    return chroma_model


def chroma_octave_modeling(act_func, hidden_dim, layer_number):
    # analysis chroma and octave info
    print('modeling chroma and octave info')
    # configuration
    chroma_num = 655

    # input layers
    input_melody = Input((25,), name='input_melody')
    input_harmony = Input((104,), name='input_harmony')

    # get info from melody, window=2
    melody_chroma = Lambda(name='melody_chroma', function=get_info, output_shape=(5,),
                        arguments={'index': 1, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_octave = Lambda(name='melody_octave', function=get_info, output_shape=(5,),
                        arguments={'index': 2, 'info_num': 5, 'total_num': 25})(input_melody)

    # get info from harmony, max length= 25+1 note events
    harmony_chroma = Lambda(get_info, name='harmony_chroma', output_shape=(26,),
                            arguments={'index': 0, 'info_num': 4, 'total_num': 104})(
        input_harmony)
    harmony_octave = Lambda(get_info, name='harmony_octave', output_shape=(26,),
                            arguments={'index': 1, 'info_num': 4, 'total_num': 104})(
        input_harmony)

    # embedding layers
    chroma_embedding = EmbeddingRet(input_dim=chroma_num + 1, output_dim=hidden_dim, mask_zero=True,
                                    name='chroma_embedding')

    # get embeddings and mask zero
    melody_chroma_rep, chroma_matirx = chroma_embedding(melody_chroma)
    melody_chroma_rep = Masking(mask_value=0.0)(melody_chroma_rep)

    harmony_chroma_rep, _ = chroma_embedding(harmony_chroma)
    harmony_chroma_rep = Masking(mask_value=0.0)(harmony_chroma_rep)

    # get octave representation
    octave_rep = Lambda(lambda x: K.mean(x, axis=0), name='octave_embedding')(chroma_matirx)
    reshape_octave = Lambda(lambda x: K.expand_dims(x), name='reshape_octave')

    # get pitch representation, namely fusing chroma and octave
    melody_octave_reshape = reshape_octave(melody_octave)
    melody_octave_rep = Lambda(lambda x: x * octave_rep, name='melody_octave_rep')(melody_octave_reshape)
    melody_octave_rep = Masking(mask_value=0.0)(melody_octave_rep)
    melody_pitch_rep = Add(name='melody_pitch_rep')([melody_chroma_rep, melody_octave_rep])
    melody_pitch_rep = BatchNormalization()(melody_pitch_rep)

    harmony_octave_reshape = reshape_octave(harmony_octave)
    harmony_octave_rep = Lambda(lambda x: x * octave_rep, name='harmony_octave_rep')(harmony_octave_reshape)
    harmony_octave_rep = Masking(mask_value=0.0)(harmony_octave_rep)
    harmony_pitch_rep = Add(name='harmony_pitch_rep')([harmony_chroma_rep, harmony_octave_rep])
    harmony_pitch_rep = Masking(mask_value=0.0)(harmony_pitch_rep)
    harmony_pitch_rep = BatchNormalization()(harmony_pitch_rep)
    # get note representation
    melody_note_rep = melody_pitch_rep
    harmony_note_rep = harmony_pitch_rep

    for i in range(layer_number):
        melody_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(melody_note_rep)
        harmony_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(harmony_note_rep)

    harmony_rep = GlobalAveragePooling1D(name='harmony_rep')(harmony_note_rep)
    harmony_rep = BatchNormalization()(harmony_rep)
    for i in range(layer_number - 1):
        melody_note_rep = Bidirectional(GRU(hidden_dim // 2, activation=act_func, return_sequences=True))(melody_note_rep)
    melody_rep = Bidirectional(GRU(hidden_dim, activation=act_func), name='melody_rep')(melody_note_rep)
    melody_rep = BatchNormalization()(melody_rep)
    # get outputs
    melody_outputs = Dense(4, activation='sigmoid', name='melody_output')(melody_rep)
    harmony_output = Dense(4, activation='sigmoid', name='harmony_output')(harmony_rep)
    label_output = Average(name='label_output')([melody_outputs, harmony_output])

    chroma_octave_model = Model(inputs=[input_melody, input_harmony], outputs=label_output)
    chroma_octave_model.summary()
    return chroma_octave_model


def chroma_position_modeling(act_func, hidden_dim, layer_number):
    # analysis chroma and position info
    print('modeling chroma and position info')
    # configuration
    chroma_num = 655

    # input layers
    input_melody = Input((25,), name='input_melody')
    input_harmony = Input((104,), name='input_harmony')

    # get info from melody, window=2
    melody_position = Lambda(name='melody_position', function=get_info, output_shape=(5,),
                          arguments={'index': 0, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_chroma = Lambda(name='melody_chroma', function=get_info, output_shape=(5,),
                        arguments={'index': 1, 'info_num': 5, 'total_num': 25})(input_melody)

    # get info from harmony, max length= 25+1 note events
    harmony_chroma = Lambda(get_info, name='harmony_chroma', output_shape=(26,),
                            arguments={'index': 0, 'info_num': 4, 'total_num': 104})(
        input_harmony)
    # embedding layers
    position_embedding = Position_Embedding(name='postion_embedding')
    chroma_embedding = EmbeddingRet(input_dim=chroma_num + 1, output_dim=hidden_dim, mask_zero=True,
                                    name='chroma_embedding')

    # get embeddings and mask zero
    melody_position_rep = position_embedding(melody_position)
    melody_chroma_rep, _ = chroma_embedding(melody_chroma)
    melody_chroma_rep = Masking(mask_value=0.0)(melody_chroma_rep)

    harmony_chroma_rep, _ = chroma_embedding(harmony_chroma)
    harmony_chroma_rep = Masking(mask_value=0.0)(harmony_chroma_rep)

    # get note representation
    melody_note_rep = Concatenate(name='melody_note_representation')(
        [melody_position_rep, melody_chroma_rep])
    harmony_note_rep = harmony_chroma_rep

    for i in range(layer_number):
        melody_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(melody_note_rep)
        harmony_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(harmony_note_rep)

    harmony_rep = GlobalAveragePooling1D(name='harmony_rep')(harmony_note_rep)
    harmony_rep = BatchNormalization()(harmony_rep)
    for i in range(layer_number - 1):
        melody_note_rep = Bidirectional(GRU(hidden_dim // 2, activation=act_func, return_sequences=True))(melody_note_rep)
    melody_rep = Bidirectional(GRU(hidden_dim, activation=act_func), name='melody_rep')(melody_note_rep)
    melody_rep = BatchNormalization()(melody_rep)
    # get outputs
    melody_outputs = Dense(4, activation='sigmoid', name='melody_output')(melody_rep)
    harmony_output = Dense(4, activation='sigmoid', name='harmony_output')(harmony_rep)
    label_output = Average(name='label_output')([melody_outputs, harmony_output])

    chroma_position_model = Model(inputs=[input_melody, input_harmony], outputs=label_output)
    chroma_position_model.summary()
    return chroma_position_model


def chroma_velocity_modeling(act_func, hidden_dim, layer_number):
    # analysis chroma and velocity info
    print('only modeling chroma and velocity info')
    # configuration
    chroma_num = 655
    velocity_num = 10
    state_num = 3

    # input layers
    input_melody = Input((25,), name='input_melody')
    input_harmony = Input((104,), name='input_harmony')

    # get info from melody, window=2
    melody_chroma = Lambda(name='melody_chroma', function=get_info, output_shape=(5,),
                        arguments={'index': 1, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_velocity = Lambda(name='melody_velocity', function=get_info, output_shape=(5,),
                          arguments={'index': 3, 'info_num': 5, 'total_num': 25})(input_melody)

    # get info from harmony, max length= 25+1 note events
    harmony_chroma = Lambda(get_info, name='harmony_chroma', output_shape=(26,),
                            arguments={'index': 0, 'info_num': 4, 'total_num': 104})(
        input_harmony)

    harmony_velocity = Lambda(get_info, name='harmony_velocity', output_shape=(26,),
                              arguments={'index': 2, 'info_num': 4, 'total_num': 104})(
        input_harmony)

    # embedding layers')
    chroma_embedding = EmbeddingRet(input_dim=chroma_num + 1, output_dim=hidden_dim, mask_zero=True,
                                    name='chroma_embedding')
    velocity_embedding = Embedding(input_dim=velocity_num + 1, output_dim=hidden_dim, mask_zero=True,
                                   name='velocity_embedding')

    # get embeddings and mask zero
    melody_chroma_rep, chroma_matirx = chroma_embedding(melody_chroma)
    melody_chroma_rep = Masking(mask_value=0.0)(melody_chroma_rep)
    melody_velocity_rep = velocity_embedding(melody_velocity)
    melody_velocity_rep = Masking(mask_value=0.0)(melody_velocity_rep)

    harmony_chroma_rep, _ = chroma_embedding(harmony_chroma)
    harmony_chroma_rep = Masking(mask_value=0.0)(harmony_chroma_rep)
    harmony_velocity_rep = velocity_embedding(harmony_velocity)
    harmony_velocity_rep = Masking(mask_value=0.0)(harmony_velocity_rep)

    # get note representation
    melody_note_rep = Concatenate(name='melody_note_representation')(
        [melody_chroma_rep, melody_velocity_rep])
    harmony_note_rep = Concatenate(name='harmony_note_representation')(
        [harmony_chroma_rep, harmony_velocity_rep])

    for i in range(layer_number):
        melody_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(melody_note_rep)
        harmony_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(harmony_note_rep)

    harmony_rep = GlobalAveragePooling1D(name='harmony_rep')(harmony_note_rep)
    harmony_rep = BatchNormalization()(harmony_rep)
    for i in range(layer_number - 1):
        melody_note_rep = Bidirectional(GRU(hidden_dim // 2, activation=act_func, return_sequences=True))(melody_note_rep)
    melody_rep = Bidirectional(GRU(hidden_dim, activation=act_func), name='melody_rep')(melody_note_rep)
    melody_rep = BatchNormalization()(melody_rep)
    # get outputs
    melody_outputs = Dense(4, activation='sigmoid', name='melody_output')(melody_rep)
    harmony_output = Dense(4, activation='sigmoid', name='harmony_output')(harmony_rep)
    label_output = Average(name='label_output')([melody_outputs, harmony_output])

    chroma_velocity_model = Model(inputs=[input_melody, input_harmony], outputs=label_output)
    chroma_velocity_model.summary()
    return chroma_velocity_model


def chroma_state_modeling(act_func, hidden_dim, layer_number):
    # analysis chroma and state info
    print('only modeling chroma and state info')
    # configuration
    chroma_num = 655
    state_num = 3

    # input layers
    input_melody = Input((25,), name='input_melody')
    input_harmony = Input((104,), name='input_harmony')

    # get info from melody, window=2
    melody_chroma = Lambda(name='melody_chroma', function=get_info, output_shape=(5,),
                        arguments={'index': 1, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_state = Lambda(name='melody_state', function=get_info, output_shape=(5,),
                       arguments={'index': 4, 'info_num': 5, 'total_num': 25})(input_melody)

    # get info from harmony, max length= 25+1 note events
    harmony_chroma = Lambda(get_info, name='harmony_chroma', output_shape=(26,),
                            arguments={'index': 0, 'info_num': 4, 'total_num': 104})(
        input_harmony)
    harmony_state = Lambda(get_info, name='harmony_state', output_shape=(26,),
                           arguments={'index': 3, 'info_num': 4, 'total_num': 104})(
        input_harmony)

    # embedding layers
    chroma_embedding = EmbeddingRet(input_dim=chroma_num + 1, output_dim=hidden_dim, mask_zero=True,
                                    name='chroma_embedding')
    state_embedding = Embedding(input_dim=state_num + 1, output_dim=hidden_dim, mask_zero=True, name='state_embedding')

    # get embeddings and mask zero
    melody_chroma_rep, chroma_matirx = chroma_embedding(melody_chroma)
    melody_chroma_rep = Masking(mask_value=0.0)(melody_chroma_rep)
    melody_state_rep = state_embedding(melody_state)
    melody_state_rep = Masking(mask_value=0.0)(melody_state_rep)

    harmony_chroma_rep, _ = chroma_embedding(harmony_chroma)
    harmony_chroma_rep = Masking(mask_value=0.0)(harmony_chroma_rep)
    harmony_state_rep = state_embedding(harmony_state)
    harmony_state_rep = Masking(mask_value=0.0)(harmony_state_rep)

    # get note representation
    melody_note_rep = Concatenate(name='melody_note_representation')(
        [melody_chroma_rep, melody_state_rep])
    harmony_note_rep = Concatenate(name='harmony_note_representation')(
        [harmony_chroma_rep, harmony_state_rep])

    for i in range(layer_number):
        melody_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(melody_note_rep)
        harmony_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(harmony_note_rep)

    harmony_rep = GlobalAveragePooling1D(name='harmony_rep')(harmony_note_rep)
    harmony_rep = BatchNormalization()(harmony_rep)
    for i in range(layer_number - 1):
        melody_note_rep = Bidirectional(GRU(hidden_dim // 2, activation=act_func, return_sequences=True))(melody_note_rep)
    melody_rep = Bidirectional(GRU(hidden_dim, activation=act_func), name='melody_rep')(melody_note_rep)
    melody_rep = BatchNormalization()(melody_rep)
    # get outputs
    melody_outputs = Dense(4, activation='sigmoid', name='melody_output')(melody_rep)
    harmony_output = Dense(4, activation='sigmoid', name='harmony_output')(harmony_rep)
    label_output = Average(name='label_output')([melody_outputs, harmony_output])

    chroma_state_model = Model(inputs=[input_melody, input_harmony], outputs=label_output)
    chroma_state_model.summary()
    return chroma_state_model


def melody_modeling(act_func, hidden_dim, layer_number):
    # analysis melody info
    print('only modeling melody info')
    # configuration
    chroma_num = 655
    velocity_num = 10
    state_num = 3

    # input layers
    input_melody = Input((25,), name='input_melody')

    # get info from melody, window=2
    melody_position = Lambda(name='melody_position', function=get_info, output_shape=(5,),
                          arguments={'index': 0, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_chroma = Lambda(name='melody_chroma', function=get_info, output_shape=(5,),
                        arguments={'index': 1, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_octave = Lambda(name='melody_octave', function=get_info, output_shape=(5,),
                        arguments={'index': 2, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_velocity = Lambda(name='melody_velocity', function=get_info, output_shape=(5,),
                          arguments={'index': 3, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_state = Lambda(name='melody_state', function=get_info, output_shape=(5,),
                       arguments={'index': 4, 'info_num': 5, 'total_num': 25})(input_melody)

    # embedding layers
    position_embedding = Position_Embedding(name='postion_embedding')
    chroma_embedding = EmbeddingRet(input_dim=chroma_num + 1, output_dim=hidden_dim, mask_zero=True,
                                    name='chroma_embedding')
    velocity_embedding = Embedding(input_dim=velocity_num + 1, output_dim=hidden_dim, mask_zero=True,
                                   name='velocity_embedding')
    state_embedding = Embedding(input_dim=state_num + 1, output_dim=hidden_dim, mask_zero=True, name='state_embedding')

    # get embeddings and mask zero
    melody_position_rep = position_embedding(melody_position)
    melody_chroma_rep, chroma_matirx = chroma_embedding(melody_chroma)
    melody_chroma_rep = Masking(mask_value=0.0)(melody_chroma_rep)
    melody_velocity_rep = velocity_embedding(melody_velocity)
    melody_velocity_rep = Masking(mask_value=0.0)(melody_velocity_rep)
    melody_state_rep = state_embedding(melody_state)
    melody_state_rep = Masking(mask_value=0.0)(melody_state_rep)

    # get octave representation
    octave_rep = Lambda(lambda x: K.mean(x, axis=0), name='octave_embedding')(chroma_matirx)
    reshape_octave = Lambda(lambda x: K.expand_dims(x), name='reshape_octave')

    # get pitch representation, namely fusing chroma and octave
    melody_octave_reshape = reshape_octave(melody_octave)
    melody_octave_rep = Lambda(lambda x: x * octave_rep, name='melody_octave_rep')(melody_octave_reshape)
    melody_octave_rep = Masking(mask_value=0.0)(melody_octave_rep)
    melody_pitch_rep = Add(name='melody_pitch_rep')([melody_chroma_rep, melody_octave_rep])
    melody_pitch_rep = BatchNormalization()(melody_pitch_rep)

    # get note representation
    melody_note_rep = Concatenate(name='melody_note_representation')(
        [melody_position_rep, melody_pitch_rep, melody_velocity_rep, melody_state_rep])
    for i in range(layer_number):
        melody_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(melody_note_rep)

    for i in range(layer_number - 1):
        melody_note_rep = Bidirectional(GRU(hidden_dim // 2, activation=act_func, return_sequences=True))(melody_note_rep)
    melody_rep = Bidirectional(GRU(hidden_dim, activation=act_func), name='melody_rep')(melody_note_rep)
    melody_rep = BatchNormalization()(melody_rep)
    # get outputs
    label_output = Dense(4, activation='sigmoid', name='melody_output')(melody_rep)

    melody_model = Model(inputs=input_melody, outputs=label_output)
    melody_model.summary()
    return melody_model


def harmony_modeling(act_func, hidden_dim, layer_number):
    # analysis melody info
    print('only modeling harmony info')
    # configuration
    chroma_num = 655
    velocity_num = 10
    state_num = 3

    # input layers
    input_harmony = Input((104,), name='input_harmony')

    # get info from harmony, max length= 25+1 note events
    harmony_chroma = Lambda(get_info, name='harmony_chroma', output_shape=(26,),
                            arguments={'index': 0, 'info_num': 4, 'total_num': 104})(
        input_harmony)
    harmony_octave = Lambda(get_info, name='harmony_octave', output_shape=(26,),
                            arguments={'index': 1, 'info_num': 4, 'total_num': 104})(
        input_harmony)
    harmony_velocity = Lambda(get_info, name='harmony_velocity', output_shape=(26,),
                              arguments={'index': 2, 'info_num': 4, 'total_num': 104})(
        input_harmony)
    harmony_state = Lambda(get_info, name='harmony_state', output_shape=(26,),
                           arguments={'index': 3, 'info_num': 4, 'total_num': 104})(
        input_harmony)

    # embedding layers
    position_embedding = Position_Embedding(name='postion_embedding')
    chroma_embedding = EmbeddingRet(input_dim=chroma_num + 1, output_dim=hidden_dim, mask_zero=True,
                                    name='chroma_embedding')
    velocity_embedding = Embedding(input_dim=velocity_num + 1, output_dim=hidden_dim, mask_zero=True,
                                   name='velocity_embedding')
    state_embedding = Embedding(input_dim=state_num + 1, output_dim=hidden_dim, mask_zero=True, name='state_embedding')

    harmony_chroma_rep, chroma_matirx = chroma_embedding(harmony_chroma)
    harmony_chroma_rep = Masking(mask_value=0.0)(harmony_chroma_rep)
    harmony_velocity_rep = velocity_embedding(harmony_velocity)
    harmony_velocity_rep = Masking(mask_value=0.0)(harmony_velocity_rep)
    harmony_state_rep = state_embedding(harmony_state)
    harmony_state_rep = Masking(mask_value=0.0)(harmony_state_rep)

    # get octave representation
    octave_rep = Lambda(lambda x: K.mean(x, axis=0), name='octave_embedding')(chroma_matirx)
    reshape_octave = Lambda(lambda x: K.expand_dims(x), name='reshape_octave')

    harmony_octave_reshape = reshape_octave(harmony_octave)
    harmony_octave_rep = Lambda(lambda x: x * octave_rep, name='harmony_octave_rep')(harmony_octave_reshape)
    harmony_octave_rep = Masking(mask_value=0.0)(harmony_octave_rep)
    harmony_pitch_rep = Add(name='harmony_pitch_rep')([harmony_chroma_rep, harmony_octave_rep])
    harmony_pitch_rep = Masking(mask_value=0.0)(harmony_pitch_rep)
    harmony_pitch_rep = BatchNormalization()(harmony_pitch_rep)
    # get note representation
    harmony_note_rep = Concatenate(name='harmony_note_representation')(
        [harmony_pitch_rep, harmony_velocity_rep, harmony_state_rep])

    for i in range(layer_number):
        harmony_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(harmony_note_rep)

    harmony_rep = GlobalAveragePooling1D(name='harmony_rep')(harmony_note_rep)
    harmony_rep = BatchNormalization()(harmony_rep)
    label_output = Dense(4, activation='sigmoid', name='harmony_output')(harmony_rep)

    harmony_model = Model(inputs=input_harmony, outputs=label_output)
    harmony_model.summary()
    return harmony_model


def chroma_melody_modeling(act_func, hidden_dim, layer_number):
    # analysis chroma info in melody
    print('modeling chroma info in melody')
    # configuration
    chroma_num = 655

    # input layers
    input_melody = Input((25,), name='input_melody')

    # get info from melody, window=2
    melody_chroma = Lambda(name='melody_chroma', function=get_info, output_shape=(5,),
                        arguments={'index': 1, 'info_num': 5, 'total_num': 25})(input_melody)

    # embedding layers
    chroma_embedding = EmbeddingRet(input_dim=chroma_num + 1, output_dim=hidden_dim, mask_zero=True,
                                    name='chroma_embedding')

    # get embeddings and mask zero
    melody_chroma_rep, _ = chroma_embedding(melody_chroma)
    melody_chroma_rep = Masking(mask_value=0.0)(melody_chroma_rep)

    # get note representation
    melody_note_rep = melody_chroma_rep

    for i in range(layer_number):
        melody_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(melody_note_rep)

    for i in range(layer_number - 1):
        melody_note_rep = Bidirectional(GRU(hidden_dim // 2, activation=act_func, return_sequences=True))(melody_note_rep)
    melody_rep = Bidirectional(GRU(hidden_dim, activation=act_func), name='melody_rep')(melody_note_rep)
    melody_rep = BatchNormalization()(melody_rep)
    # get outputs
    melody_outputs = Dense(4, activation='sigmoid', name='melody_output')(melody_rep)

    chroma_melody_model = Model(inputs=input_melody, outputs=melody_outputs)
    chroma_melody_model.summary()
    return chroma_melody_model


def chroma_harmony_modeling(act_func, hidden_dim, layer_number):
    # analysis chroma info in harmony
    print('modeling chroma info in harmony')
    # configuration
    chroma_num = 655

    # input layers
    input_harmony = Input((104,), name='input_harmony')

    # get info from harmony, max length= 25+1 note events
    harmony_chroma = Lambda(get_info, name='harmony_chroma', output_shape=(26,),
                            arguments={'index': 0, 'info_num': 4, 'total_num': 104})(
        input_harmony)

    # embedding layers
    chroma_embedding = EmbeddingRet(input_dim=chroma_num + 1, output_dim=hidden_dim, mask_zero=True,
                                    name='chroma_embedding')

    # get embeddings and mask zero

    harmony_chroma_rep, _ = chroma_embedding(harmony_chroma)
    harmony_chroma_rep = Masking(mask_value=0.0)(harmony_chroma_rep)

    # get note representation
    harmony_note_rep = harmony_chroma_rep

    for i in range(layer_number):
        harmony_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(harmony_note_rep)

    harmony_rep = GlobalAveragePooling1D(name='harmony_rep')(harmony_note_rep)
    harmony_rep = BatchNormalization()(harmony_rep)

    # get outputs
    harmony_output = Dense(4, activation='sigmoid', name='harmony_output')(harmony_rep)

    chroma_harmony_model = Model(inputs=input_harmony, outputs=harmony_output)
    chroma_harmony_model.summary()
    return chroma_harmony_model


def chroma_state_melody_modeling(act_func, hidden_dim, layer_number):
    # analysis chroma and state info in melody
    print('only modeling chroma and state info in melody')
    # configuration
    chroma_num = 655
    state_num = 3

    # input layers
    input_melody = Input((25,), name='input_melody')

    # get info from melody, window=2
    melody_chroma = Lambda(name='melody_chroma', function=get_info, output_shape=(5,),
                        arguments={'index': 1, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_state = Lambda(name='melody_state', function=get_info, output_shape=(5,),
                       arguments={'index': 4, 'info_num': 5, 'total_num': 25})(input_melody)

    # embedding layers
    chroma_embedding = EmbeddingRet(input_dim=chroma_num + 1, output_dim=hidden_dim, mask_zero=True,
                                    name='chroma_embedding')
    state_embedding = Embedding(input_dim=state_num + 1, output_dim=hidden_dim, mask_zero=True, name='state_embedding')

    # get embeddings and mask zero
    melody_chroma_rep, chroma_matirx = chroma_embedding(melody_chroma)
    melody_chroma_rep = Masking(mask_value=0.0)(melody_chroma_rep)
    melody_state_rep = state_embedding(melody_state)
    melody_state_rep = Masking(mask_value=0.0)(melody_state_rep)

    # get note representation
    melody_note_rep = Concatenate(name='melody_note_representation')(
        [melody_chroma_rep, melody_state_rep])

    for i in range(layer_number):
        melody_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(melody_note_rep)

    for i in range(layer_number - 1):
        melody_note_rep = Bidirectional(GRU(hidden_dim // 2, activation=act_func, return_sequences=True))(melody_note_rep)
    melody_rep = Bidirectional(GRU(hidden_dim, activation=act_func), name='melody_rep')(melody_note_rep)
    melody_rep = BatchNormalization()(melody_rep)
    # get outputs
    melody_outputs = Dense(4, activation='sigmoid', name='melody_output')(melody_rep)

    chroma_state_melody_model = Model(inputs=input_melody, outputs=melody_outputs)
    chroma_state_melody_model.summary()
    return chroma_state_melody_model


def chroma_state_harmony_modeling(act_func, hidden_dim, layer_number):
    # analysis chroma and state info in harmony
    print('only modeling chroma and state info in harmony')
    # configuration
    chroma_num = 655
    state_num = 3

    # input layers
    input_harmony = Input((104,), name='input_harmony')

    # get info from harmony, max length= 25+1 note events
    harmony_chroma = Lambda(get_info, name='harmony_chroma', output_shape=(26,),
                            arguments={'index': 0, 'info_num': 4, 'total_num': 104})(
        input_harmony)
    harmony_state = Lambda(get_info, name='harmony_state', output_shape=(26,),
                           arguments={'index': 3, 'info_num': 4, 'total_num': 104})(
        input_harmony)

    # embedding layers
    chroma_embedding = EmbeddingRet(input_dim=chroma_num + 1, output_dim=hidden_dim, mask_zero=True,
                                    name='chroma_embedding')
    state_embedding = Embedding(input_dim=state_num + 1, output_dim=hidden_dim, mask_zero=True, name='state_embedding')

    # get embeddings and mask zero

    harmony_chroma_rep, _ = chroma_embedding(harmony_chroma)
    harmony_chroma_rep = Masking(mask_value=0.0)(harmony_chroma_rep)
    harmony_state_rep = state_embedding(harmony_state)
    harmony_state_rep = Masking(mask_value=0.0)(harmony_state_rep)

    # get note representation
    harmony_note_rep = Concatenate(name='harmony_note_representation')(
        [harmony_chroma_rep, harmony_state_rep])

    for i in range(layer_number):
        harmony_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(harmony_note_rep)

    harmony_rep = GlobalAveragePooling1D(name='harmony_rep')(harmony_note_rep)
    harmony_rep = BatchNormalization()(harmony_rep)

    # get outputs
    harmony_output = Dense(4, activation='sigmoid', name='harmony_output')(harmony_rep)

    chroma_state_harmony_model = Model(inputs=input_harmony, outputs=harmony_output)
    chroma_state_harmony_model.summary()
    return chroma_state_harmony_model


def chroma_octave_melody_modeling(act_func, hidden_dim, layer_number):
    # analysis chroma and octave info in melody
    print('modeling chroma and octave info in melody')
    # configuration
    chroma_num = 655

    # input layers
    input_melody = Input((25,), name='input_melody')

    # get info from melody, window=2
    melody_chroma = Lambda(name='melody_chroma', function=get_info, output_shape=(5,),
                        arguments={'index': 1, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_octave = Lambda(name='melody_octave', function=get_info, output_shape=(5,),
                        arguments={'index': 2, 'info_num': 5, 'total_num': 25})(input_melody)

    # embedding layers
    chroma_embedding = EmbeddingRet(input_dim=chroma_num + 1, output_dim=hidden_dim, mask_zero=True,
                                    name='chroma_embedding')

    # get embeddings and mask zero
    melody_chroma_rep, chroma_matirx = chroma_embedding(melody_chroma)
    melody_chroma_rep = Masking(mask_value=0.0)(melody_chroma_rep)

    # get octave representation
    octave_rep = Lambda(lambda x: K.mean(x, axis=0), name='octave_embedding')(chroma_matirx)
    reshape_octave = Lambda(lambda x: K.expand_dims(x), name='reshape_octave')

    # get pitch representation, namely fusing chroma and octave
    melody_octave_reshape = reshape_octave(melody_octave)
    melody_octave_rep = Lambda(lambda x: x * octave_rep, name='melody_octave_rep')(melody_octave_reshape)
    melody_octave_rep = Masking(mask_value=0.0)(melody_octave_rep)
    melody_pitch_rep = Add(name='melody_pitch_rep')([melody_chroma_rep, melody_octave_rep])
    melody_pitch_rep = BatchNormalization()(melody_pitch_rep)

    # get note representation
    melody_note_rep = melody_pitch_rep

    for i in range(layer_number):
        melody_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(melody_note_rep)

    for i in range(layer_number - 1):
        melody_note_rep = Bidirectional(GRU(hidden_dim // 2, activation=act_func, return_sequences=True))(melody_note_rep)
    melody_rep = Bidirectional(GRU(hidden_dim, activation=act_func), name='melody_rep')(melody_note_rep)
    melody_rep = BatchNormalization()(melody_rep)
    # get outputs
    melody_outputs = Dense(4, activation='sigmoid', name='melody_output')(melody_rep)

    chroma_octave_melody_model = Model(inputs=input_melody, outputs=melody_outputs)
    chroma_octave_melody_model.summary()
    return chroma_octave_melody_model


def chroma_octave_harmony_modeling(act_func, hidden_dim, layer_number):
    # analysis chroma and octave info in harmony
    print('modeling chroma and octave info in harmony')
    # configuration
    chroma_num = 655

    # input layers
    input_harmony = Input((104,), name='input_harmony')


    # get info from harmony, max length= 25+1 note events
    harmony_chroma = Lambda(get_info, name='harmony_chroma', output_shape=(26,),
                            arguments={'index': 0, 'info_num': 4, 'total_num': 104})(
        input_harmony)
    harmony_octave = Lambda(get_info, name='harmony_octave', output_shape=(26,),
                            arguments={'index': 1, 'info_num': 4, 'total_num': 104})(
        input_harmony)

    # embedding layers
    chroma_embedding = EmbeddingRet(input_dim=chroma_num + 1, output_dim=hidden_dim, mask_zero=True,
                                    name='chroma_embedding')


    harmony_chroma_rep, chroma_matirx = chroma_embedding(harmony_chroma)
    harmony_chroma_rep = Masking(mask_value=0.0)(harmony_chroma_rep)

    # get octave representation
    octave_rep = Lambda(lambda x: K.mean(x, axis=0), name='octave_embedding')(chroma_matirx)
    reshape_octave = Lambda(lambda x: K.expand_dims(x), name='reshape_octave')

    # get pitch representation, namely fusing chroma and octave

    harmony_octave_reshape = reshape_octave(harmony_octave)
    harmony_octave_rep = Lambda(lambda x: x * octave_rep, name='harmony_octave_rep')(harmony_octave_reshape)
    harmony_octave_rep = Masking(mask_value=0.0)(harmony_octave_rep)
    harmony_pitch_rep = Add(name='harmony_pitch_rep')([harmony_chroma_rep, harmony_octave_rep])
    harmony_pitch_rep = Masking(mask_value=0.0)(harmony_pitch_rep)
    harmony_pitch_rep = BatchNormalization()(harmony_pitch_rep)
    # get note representation
    harmony_note_rep = harmony_pitch_rep

    for i in range(layer_number):
        harmony_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(harmony_note_rep)

    harmony_rep = GlobalAveragePooling1D(name='harmony_rep')(harmony_note_rep)
    harmony_rep = BatchNormalization()(harmony_rep)

    # get outputs
    harmony_output = Dense(4, activation='sigmoid', name='harmony_output')(harmony_rep)

    chroma_octave_harmony_model = Model(inputs=input_harmony, outputs=harmony_output)
    chroma_octave_harmony_model.summary()
    return chroma_octave_harmony_model


def chroma_position_melody_modeling(act_func, hidden_dim, layer_number):
    # analysis chroma and position info in melody
    print('modeling chroma and position info in melody')
    # configuration
    chroma_num = 655

    # input layers
    input_melody = Input((25,), name='input_melody')

    # get info from melody, window=2
    melody_position = Lambda(name='melody_position', function=get_info, output_shape=(5,),
                          arguments={'index': 0, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_chroma = Lambda(name='melody_chroma', function=get_info, output_shape=(5,),
                        arguments={'index': 1, 'info_num': 5, 'total_num': 25})(input_melody)


    # embedding layers
    position_embedding = Position_Embedding(name='postion_embedding')
    chroma_embedding = EmbeddingRet(input_dim=chroma_num + 1, output_dim=hidden_dim, mask_zero=True,
                                    name='chroma_embedding')

    # get embeddings and mask zero
    melody_position_rep = position_embedding(melody_position)
    melody_chroma_rep, _ = chroma_embedding(melody_chroma)
    melody_chroma_rep = Masking(mask_value=0.0)(melody_chroma_rep)


    # get note representation
    melody_note_rep = Concatenate(name='melody_note_representation')(
        [melody_position_rep, melody_chroma_rep])

    for i in range(layer_number):
        melody_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(melody_note_rep)

    for i in range(layer_number - 1):
        melody_note_rep = Bidirectional(GRU(hidden_dim // 2, activation=act_func, return_sequences=True))(melody_note_rep)
    melody_rep = Bidirectional(GRU(hidden_dim, activation=act_func), name='melody_rep')(melody_note_rep)
    melody_rep = BatchNormalization()(melody_rep)
    # get outputs
    melody_outputs = Dense(4, activation='sigmoid', name='melody_output')(melody_rep)

    chroma_position_melody_model = Model(inputs=input_melody, outputs=melody_outputs)
    chroma_position_melody_model.summary()
    return chroma_position_melody_model


def chroma_velocity_melody_modeling(act_func, hidden_dim, layer_number):
    # analysis chroma and velocity info in melody
    print('only modeling chroma and velocity info in melody')
    # configuration
    chroma_num = 655
    velocity_num = 10

    # input layers
    input_melody = Input((25,), name='input_melody')

    # get info from melody, window=2
    melody_chroma = Lambda(name='melody_chroma', function=get_info, output_shape=(5,),
                        arguments={'index': 1, 'info_num': 5, 'total_num': 25})(input_melody)
    melody_velocity = Lambda(name='melody_velocity', function=get_info, output_shape=(5,),
                          arguments={'index': 3, 'info_num': 5, 'total_num': 25})(input_melody)



    # embedding layers')
    chroma_embedding = EmbeddingRet(input_dim=chroma_num + 1, output_dim=hidden_dim, mask_zero=True,
                                    name='chroma_embedding')
    velocity_embedding = Embedding(input_dim=velocity_num + 1, output_dim=hidden_dim, mask_zero=True,
                                   name='velocity_embedding')

    # get embeddings and mask zero
    melody_chroma_rep, chroma_matirx = chroma_embedding(melody_chroma)
    melody_chroma_rep = Masking(mask_value=0.0)(melody_chroma_rep)
    melody_velocity_rep = velocity_embedding(melody_velocity)
    melody_velocity_rep = Masking(mask_value=0.0)(melody_velocity_rep)


    # get note representation
    melody_note_rep = Concatenate(name='melody_note_representation')(
        [melody_chroma_rep, melody_velocity_rep])


    for i in range(layer_number):
        melody_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(melody_note_rep)

    for i in range(layer_number - 1):
        melody_note_rep = Bidirectional(GRU(hidden_dim // 2, activation=act_func, return_sequences=True))(melody_note_rep)
    melody_rep = Bidirectional(GRU(hidden_dim, activation=act_func), name='melody_rep')(melody_note_rep)
    melody_rep = BatchNormalization()(melody_rep)
    # get outputs
    melody_outputs = Dense(4, activation='sigmoid', name='melody_output')(melody_rep)

    chroma_velocity_melody_model = Model(inputs=input_melody, outputs=melody_outputs)
    chroma_velocity_melody_model.summary()
    return chroma_velocity_melody_model


def chroma_velocity_harmony_modeling(act_func, hidden_dim, layer_number):
    # analysis chroma and velocity info in harmony
    print('only modeling chroma and velocity info in harmony')
    # configuration
    chroma_num = 655
    velocity_num = 10

    # input layers
    input_harmony = Input((104,), name='input_harmony')


    # get info from harmony, max length= 25+1 note events
    harmony_chroma = Lambda(get_info, name='harmony_chroma', output_shape=(26,),
                            arguments={'index': 0, 'info_num': 4, 'total_num': 104})(
        input_harmony)

    harmony_velocity = Lambda(get_info, name='harmony_velocity', output_shape=(26,),
                              arguments={'index': 2, 'info_num': 4, 'total_num': 104})(
        input_harmony)

    # embedding layers')
    chroma_embedding = EmbeddingRet(input_dim=chroma_num + 1, output_dim=hidden_dim, mask_zero=True,
                                    name='chroma_embedding')
    velocity_embedding = Embedding(input_dim=velocity_num + 1, output_dim=hidden_dim, mask_zero=True,
                                   name='velocity_embedding')

    # get embeddings and mask zero


    harmony_chroma_rep, _ = chroma_embedding(harmony_chroma)
    harmony_chroma_rep = Masking(mask_value=0.0)(harmony_chroma_rep)
    harmony_velocity_rep = velocity_embedding(harmony_velocity)
    harmony_velocity_rep = Masking(mask_value=0.0)(harmony_velocity_rep)

    # get note representation

    harmony_note_rep = Concatenate(name='harmony_note_representation')(
        [harmony_chroma_rep, harmony_velocity_rep])

    for i in range(layer_number):

        harmony_note_rep = TimeDistributed(Dense(hidden_dim, activation=act_func))(harmony_note_rep)

    harmony_rep = GlobalAveragePooling1D(name='harmony_rep')(harmony_note_rep)
    harmony_rep = BatchNormalization()(harmony_rep)


    # get outputs

    harmony_output = Dense(4, activation='sigmoid', name='harmony_output')(harmony_rep)

    chroma_velocity_model = Model(inputs=input_harmony, outputs=harmony_output)
    chroma_velocity_model.summary()
    return chroma_velocity_model
