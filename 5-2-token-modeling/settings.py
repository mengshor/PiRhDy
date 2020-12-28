from functions import token_modeling, generate_batch_data, token_modeling_con
from functions import chroma_modeling, chroma_octave_modeling, chroma_position_modeling, chroma_velocity_modeling, \
    chroma_state_modeling
from functions import melody_modeling, chroma_melody_modeling, chroma_state_melody_modeling, chroma_octave_melody_modeling, chroma_velocity_melody_modeling, chroma_position_melody_modeling, generate_melody_batch_data
from functions import harmony_modeling, chroma_harmony_modeling, chroma_state_harmony_modeling, chroma_octave_harmony_modeling, chroma_velocity_harmony_modeling, generate_harmony_batch_data
from layers import TimeHistory
import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os


def run_script(gpu_names=None, model_name='overall', ratio=1):
    # model configurations
    tf.compat.v1.disable_eager_execution()
    tf.keras.backend.clear_session()
    # tf.config.optimizer.set_jit(False)  # Start with XLA disabled.
    gpu_num = 0
    if gpu_names is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_names
        gpu_num = len(gpu_names.strip("\"").split(','))
    act_func = 'tanh'
    hidden_dim = 256
    batch_size = 20000
    layer_number = 2
    epoch_num = 2
    train_num = 1218504295
    test_num = 135400000

    # dataset paths
    train_path = 'token_dataset/train'
    test_path = 'token_dataset/test'

    # model settings
    model_name_list = {"overall": token_modeling,
                       "overall_con": token_modeling_con,
                       "chroma": chroma_modeling,
                       "chroma_octave": chroma_octave_modeling,
                       "chroma_position": chroma_position_modeling,
                       "chroma_velocity": chroma_velocity_modeling,
                       "chroma_state": chroma_state_modeling,
                       "melody": melody_modeling,
                       "harmony": harmony_modeling,
                       "chroma_melody": chroma_melody_modeling,
                       "chroma_harmony": chroma_harmony_modeling,
                       "chroma_state_melody": chroma_state_melody_modeling,
                       "chroma_state_harmony": chroma_state_harmony_modeling,
                       "chroma_octave_melody": chroma_octave_melody_modeling,
                       "chroma_octave_harmony": chroma_octave_harmony_modeling,
                       "chroma_position_melody": chroma_position_melody_modeling,
                       "chroma_velocity_melody": chroma_velocity_melody_modeling,
                       "chroma_velocity_harmony": chroma_velocity_harmony_modeling
                       }

    model = model_name_list[model_name](act_func, hidden_dim, layer_number)
    print('finish load ' + model_name)

    if gpu_num >= 2:
        parallel_model = multi_gpu_model(model, gpus=gpu_num)
    else:
        parallel_model = model

    # callback settings
    model_path = "best_models/"
    filepath = "/best"
    my_checkpoint = ModelCheckpoint(model_path + model_name+'_' + str(ratio) + filepath,
                                    monitor='test_acc',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='max',
                                    save_freq='epoch')
    my_earlystop = EarlyStopping(monitor='test_loss', min_delta=0, patience=10)
    time_callback = TimeHistory()
    # compile settings
    parallel_model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['acc'],
                           experimental_run_tf_function=False,
                           )

    # fit_generator settings
    generator_names = {"overall": generate_batch_data,
                       "overall_con":generate_batch_data,
                       "chroma": generate_batch_data,
                       "chroma_octave": generate_batch_data,
                       "chroma_position": generate_batch_data,
                       "chroma_velocity": generate_batch_data,
                       "chroma_state": generate_batch_data,
                       "melody": generate_melody_batch_data,
                       "harmony": generate_harmony_batch_data,
                       "chroma_melody": generate_melody_batch_data,
                       "chroma_harmony": generate_harmony_batch_data,
                       "chroma_state_melody": generate_melody_batch_data,
                       "chroma_state_harmony": generate_harmony_batch_data,
                       "chroma_octave_melody": generate_melody_batch_data,
                       "chroma_octave_harmony": generate_harmony_batch_data,
                       "chroma_position_melody": generate_melody_batch_data,
                       "chroma_velocity_melody": generate_melody_batch_data,
                       "chroma_velocity_harmony": generate_harmony_batch_data
                       }
    train_generator = generator_names[model_name](batch_size, train_path)
    test_generator = generator_names[model_name](batch_size, test_path)
    try:
        parallel_model.fit(train_generator,
                           epochs=epoch_num,
                           steps_per_epoch=np.ceil(train_num / batch_size * float(ratio)/100.0),
                           validation_data=test_generator,
                           validation_steps=np.ceil(test_num / batch_size),
                           callbacks=[my_checkpoint, my_earlystop, time_callback]
                           )
        print(time_callback.times)
    except Exception as e:
        print(e)
        print("Error model name")
