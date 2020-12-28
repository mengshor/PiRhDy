import pickle

from sklearn.model_selection import StratifiedKFold

from functions import next_phrase, acc_ass, genre_class
from functions import rank, f1
from functions import phrase_generator

import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



def genre_classification(task_name='genre_a', phrase_model='next'):
    if task_name == 'genre_a':
        data_path = 'dataset/genre/topmagd'
        num_class = 13
        class_names = ['Pop_Rock',
                       'International',
                       'Country',
                       'Electronic',
                       'Vocal',
                       'Rap',
                       'RnB',
                       'Latin',
                       'New Age',
                       'Jazz',
                       'Folk',
                       'Reggae',
                       'Blues'
                       ]
    else:
        data_path = 'dataset/genre/masd'
        num_class = 25
        class_names = ['Rock_Contemporary',
                       'Pop_Contemporary',
                       'Rock_Neo_Psychedelia',
                       'Rock_Hard',
                       'Metal_Death',
                       'Country_Traditional',
                       'Hip_Hop_Rap',
                       'Dance',
                       'Metal_Alternative',
                       'Rock_College',
                       'Pop_Latin',
                       'Pop_Indie',
                       'Gospel',
                       'Folk_International',
                       'Jazz_Classic',
                       'Electronica',
                       'Blues_Contemporary',
                       'Rock_Alternative',
                       'Big_Band',
                       'Experimental',
                       'Grunge_Emo',
                       'Punk',
                       'Metal_Heavy',
                       'RnB_Soul',
                       'Reggae'
                       ]
    temp_class = []
    for class_name in class_names:
        temp_class.append([class_name])
    data = pickle.load(open(data_path, 'rb'))
    data_num = len(data)
    files = [key for key in data.keys()]
    genres = [list(set(value)) for value in data.values()]
    for genre_name in genres:
        if genre_name not in temp_class:
            temp_class.append(genre_name)
    labels = np.zeros((data_num, num_class))
    temp_labels = []
    for idx, entry in enumerate(genres):
        for item in entry:
            labels[idx][class_names.index(item)] = 1
        temp_labels.append([temp_class.index(entry)])
    temp_labels = np.stack(temp_labels, axis=0)

    features = []
    for file in files:
        if phrase_model is not None:
            feature = np.load('features/song_dataset_{}/{}/{}.npy'.format(phrase_model, file[0], file))
        else:
            feature = np.load('dataset/song_dataset/{}/{}.npy'.format(file[0], file))
        features.append(feature)
    features = np.stack(features, axis=0)
    features = np.squeeze(features)
    n_split = 5
    model_path = 'exp_models/{}/'.format(task_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    results_file = open('exp_results/genre/{}_{}.csv'.format(task_name, phrase_model), 'a')
    for turn_id in range(10):
        split_id = 0
        print('turn {}\n'.format(turn_id))
        results_file.write('turn {}'.format(turn_id))
        for train_index, test_index in StratifiedKFold(n_split, shuffle=True).split(features, temp_labels):
            x_train, x_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            genre_info = genre_class(class_num=num_class, phrase_model=phrase_model)
            # genre_info.print_settings()
            model = genre_info.encoder()
            model.compile(optimizer='rmsprop',
                          loss='binary_crossentropy',
                          metrics=[tf.keras.metrics.AUC(), f1, 'accuracy'],
                          experimental_run_tf_function=False)
            my_checkpoint = ModelCheckpoint(
                '{}weights.{}-{}.h5'.format(model_path, turn_id, split_id),
                monitor='val_f1',
                save_best_only=True,
                save_weights_only=True,
                save_freq='epoch',
                mode='max')
            my_earlystop = EarlyStopping(monitor='val_f1',
                                         min_delta=0.0001,
                                         patience=5,
                                         mode='max')
            history = model.fit(x_train, y_train,
                                batch_size=None,
                                epochs=100,
                                validation_data=[x_test, y_test],
                                shuffle=True,
                                verbose=2,
                                callbacks=[my_checkpoint, my_earlystop])
            print('split {}'.format(split_id))
            results_file.write('split {}\n'.format(split_id))
            for key in history.history.keys():
                if 'val' in key:
                    if 'loss' not in key:
                        results_file.write('{}'.format(key))
                        for item in history.history[key]:
                            results_file.write(', {}'.format(item))
                        results_file.write('\n')
            results_file.write('\n')
            split_id += 1
        results_file.write('\n')
        results_file.flush()
    results_file.close()


def run_exp(gpu_names=None,
            task_name='next',
            ratio=100,
            epoch_num=100,
            batch_size=2000,
            phrase_model=None):
    # tensorflow setting
    tf.compat.v1.disable_eager_execution()
    tf.keras.backend.clear_session()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # gpu setting
    gpu_num = 0
    if gpu_names is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_names
        gpu_num = len(gpu_names.strip("\"").split(','))

    if task_name == 'genre_a' or task_name == 'genre_s':
        genre_classification(task_name, phrase_model)

    else:
        # train_num, test_num
        task_statistics = {'next': [3569888, 9963500],
                           'acc': [15781108, 10015000]
                           }

        train_num = int(task_statistics[task_name][0] * ratio / 100)
        test_num = int(task_statistics[task_name][1] * ratio / 100)
        print('=========================statistics======================')
        print('train_num = {}'.format(train_num))
        print('test_num = {}'.format(test_num))

        # model configuration
        task_configuration = {'next': next_phrase,
                              'acc': acc_ass
                              }
        model_info = task_configuration[task_name](phrase_model=phrase_model)
        model_info.print_settings()

        model = model_info.encoder()


        if gpu_num >= 2:
            parallel_model = multi_gpu_model(model, gpus=gpu_num)
        else:
            parallel_model = model

        # compile settings
        compile_configuration = {'next': ['binary_crossentropy',
                                          ['binary_accuracy']],
                                 'acc': ['binary_crossentropy',
                                         ['binary_accuracy']]
                                 }
        loss_function = compile_configuration[task_name][0]
        metrics_list = compile_configuration[task_name][1]

        parallel_model.compile(optimizer='rmsprop',
                               loss=loss_function,
                               metrics=metrics_list,
                               experimental_run_tf_function=False)

        # data path
        if phrase_model is not None:
            data_paths = {'next': 'features/context_next/',
                          'acc': 'features/context_acc/'
                          }
        else:
            data_paths = {'next': 'dataset/context_next/',
                          'acc': 'dataset/context_acc/'
                          }
        data_path = data_paths[task_name]

        # data generator
        generator_configuration = {'next': phrase_generator,
                                   'acc': phrase_generator
                                   }

        batch_generator = generator_configuration[task_name](batch_size, data_path, phrase_model)

        train_generator = batch_generator.train()
        train_steps = np.ceil(train_num / batch_size)
        test_generator = batch_generator.test()
        test_steps = np.ceil(test_num / batch_size)
        if phrase_model is not None:
            model_path = 'exp_models/{}_{}/'.format(task_name, phrase_model)
            result_path = 'exp_results/{}_{}/'.format(task_name, phrase_model)
        else:
            model_path = 'train_models/{}/'.format(task_name)
            result_path = 'train_results/{}/'.format(task_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        my_checkpoint = ModelCheckpoint(model_path + 'check.h5',
                                        monitor=metrics_list[0],
                                        save_best_only=True,
                                        save_weights_only=True,
                                        save_freq='epoch')
        my_earlystop = EarlyStopping(monitor='loss',
                                     min_delta=0.0001,
                                     patience=2)

        print("===================load test labels=======================")
        labels = np.loadtxt(data_path + 'test_label', delimiter='\n', max_rows=test_num)
        print("================finish load test labels===================")

        evaluate_configuration = {'next': [rank, None],
                                  'acc': [rank, None]
                                  }
        eval_func = evaluate_configuration[task_name][0]
        eval_pm = evaluate_configuration[task_name][1]
        # epoch_check = 5
        for id_e in range(5):
        # print('=================step {}==============='.format(id_e))
            parallel_model.fit(train_generator,
                               epochs=epoch_num,
                               shuffle=True,
                               steps_per_epoch=train_steps,
                               callbacks=[my_earlystop, my_checkpoint],
                               workers=1,
                               use_multiprocessing=False,
                               verbose=2
                               )
            parallel_model.save_weights(model_path + '.h5')
            scores_stop = parallel_model.predict(test_generator,
                                                 steps=test_steps,
                                                 verbose=1,
                                                 workers=1,
                                                 use_multiprocessing=False
                                                 )
            np.savetxt(result_path+ str(id_e), scores_stop)
            eval_func(y_true=labels, y_pred=scores_stop).compute(eval_pm)
