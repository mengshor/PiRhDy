import pickle
import numpy as np
import random
import os


def get_file_path(root_path, file_list, dir_list):
    # get all file names and dir names in this folder
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # get file or dir name
        dir_file_path = os.path.join(root_path, dir_file)
        # is a file or a dir
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            # get all paths
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)


# get all file names
def write_filenames(file_path, names, name_path, use_names):
    if use_names:
        for name in names:
            print(name)
            # root path
            root_path = file_path + '{}'.format(name)
            file_list = []
            dir_list = []
            file_name = open(name_path + '{}'.format(name), "wb")
            get_file_path(root_path, file_list, dir_list)
            pickle.dump(file_list, file_name)
            file_name.close()
    else:
        file_list = []
        dir_list = []
        file_name = open(name_path, "wb")
        get_file_path(file_path, file_list, dir_list)
        pickle.dump(file_list, file_name)
        file_name.close()

def check_phrase(matrix, threshold_value):
    count_num = len(np.nonzero(matrix)[0]) // 4
    if count_num / matrix.shape[1] >= threshold_value:
        return True
    return False


def pad_sample(matrix):
    matrix[0, :][matrix[0, :] == 0] = 656
    matrix[1, :][matrix[1, :] == 0] = 12
    matrix[2, :][matrix[2, :] == 0] = 11
    matrix[3, :][matrix[3, :] == 0] = 4
    padded_matrix = np.zeros([4, 128], dtype=int)
    padded_matrix[:, 0:matrix.shape[1]] = matrix

    return padded_matrix


def remove_ids(idx, end):
    scale = list(range(0, end))
    for id_info in idx:
        scale.remove(int(id_info))
    return scale


def generate_accompaniment(name):
    file_list = pickle.load(open('/filenames/phrase/{}'.format(name), 'rb'))
    #    file_list = file_list[0:100]
    train = open('context_acc/train_{}'.format(name), 'ab')
    test = open('context_acc/test_{}'.format(name), 'ab')
    train_melody_harmony = []
    test_melody_harmony = []
    acc_num_list = []
    print("data handling")
    for file_id, file in enumerate(file_list):
        #       print(file_id)
        phrases = np.load(file, allow_pickle=True)
        phrase_id = 0
        if file_id % 10 == 0:
            while str(phrase_id) in phrases:
                acc_num = 0
                phrase = phrases[str(phrase_id)]
                melody = phrase[0:4, :]
                track_num = phrase.shape[0] // 4
                for track_id in range(1, track_num):
                    accompaniment = phrase[0 + track_id * 4:4 * (track_id + 1), :]
                    if check_phrase(accompaniment, 0.5):
                        test_melody_harmony.append([file_id, melody, accompaniment])
                        acc_num += 1
                if acc_num != 0:
                    acc_num_list.append(acc_num)
                phrase_id += 1

        else:
            while str(phrase_id) in phrases:
                phrase = phrases[str(phrase_id)]
                melody = phrase[0:4, :]
                track_num = phrase.shape[0] // 4
                for track_id in range(1, track_num):
                    accompaniment = phrase[0 + track_id * 4:4 * (track_id + 1), :]
                    if check_phrase(accompaniment, 0.5):
                        train_melody_harmony.append([file_id, melody, accompaniment])
                phrase_id += 1
    # get training dataset
    print("get train")
    train_num = len(train_melody_harmony)

    for mh_id, mh in enumerate(train_melody_harmony):
        positive_song_id = mh[0]
        positive_melody = mh[1]
        positive_accompaniment = mh[2]

        positive_sample = np.concatenate(
            (pad_sample(positive_melody).flatten(), pad_sample(positive_accompaniment).flatten())
            )
        positive_sample = np.append(positive_sample, 1)
        np.savetxt(train, [positive_sample], delimiter=',', fmt='%u')

        remove_list = [mh_id]
        negative_id = random.choice(remove_ids(remove_list, train_num))
        negative_song_id = train_melody_harmony[negative_id][0]
        negative_accompaniment = train_melody_harmony[negative_id][2]
        remove_list.append(negative_id)
        while (positive_song_id == negative_song_id) or (
                np.array_equal(negative_accompaniment, positive_accompaniment)):
            negative_id = random.choice(remove_ids(remove_list, train_num))
            negative_song_id = train_melody_harmony[negative_id][0]
            negative_accompaniment = train_melody_harmony[negative_id][2]
            remove_list.append(negative_id)
        negative_sample = np.concatenate((pad_sample(positive_melody).flatten(),
                                          pad_sample(negative_accompaniment).flatten())
                                         )
        negative_sample = np.append(negative_sample, 0)
        np.savetxt(train, [negative_sample], delimiter=',', fmt='%u')
    train.close()
    # get test dataset
    print("get test")
    passed_num = 0
    test_num = len(test_melody_harmony)
    for idx, num in enumerate(acc_num_list):
        negative_num = 50 - num
        remove_list = []
        for i in range(num):
            pos_id = passed_num + i
            remove_list.append(pos_id)
            positive_sample = np.concatenate((pad_sample(test_melody_harmony[pos_id][1]).flatten(),
                                              pad_sample(test_melody_harmony[pos_id][2]).flatten())
                                             )
            positive_sample = np.append(positive_sample, 1)
            np.savetxt(test, [positive_sample], delimiter=',', fmt='%u')
        for j in range(negative_num):
            negative_id = random.choice(remove_ids(remove_list, test_num))
            negative_accompaniment = test_melody_harmony[negative_id][2]
            remove_list.append(negative_id)
            while np.array_equal(negative_accompaniment, test_melody_harmony[pos_id][2]):
                negative_id = random.choice(remove_ids(remove_list, test_num))
                negative_accompaniment = test_melody_harmony[negative_id][2]
                remove_list.append(negative_id)
            negative_sample = np.concatenate((pad_sample(test_melody_harmony[pos_id][1]).flatten(),
                                              pad_sample(negative_accompaniment).flatten())
                                             )
            negative_sample = np.append(negative_sample, 0)
            np.savetxt(test, [negative_sample], delimiter=',', fmt='%u')
        passed_num += num
    test.close()


def generate_next_phrase_melody(name):
    file_list = pickle.load(open('filenames/{}'.format(name), 'rb'))
    train = open('context_next/train_{}'.format(name), 'ab')
    test = open('context_next/test_{}'.format(name), 'ab')

    periods_train = []
    periods_test = []
    for file_id, file in enumerate(file_list):

        idx = 0
        phrases = np.load(file, allow_pickle=True)
        while str(idx + 1) in phrases:
            former = phrases[str(idx)]
            latter = phrases[str(idx + 1)]
            # only consider melody track
            melody_period = [former[0:4, :], latter[0:4, :]]
            # split train and test by song, 9:1
            if file_id % 10 == 0:
                periods_test.append([file_id, melody_period])
            else:
                periods_train.append([file_id, melody_period])
            idx += 1

    # generating train dataset, 1:1
    print("get train")
    train_num = len(periods_train)
    for idx, period in enumerate(periods_train):

        # get positive sample
        positive_song_id = period[0]
        positive_period_info = period[1]
        positive_sample = np.concatenate(
            (pad_sample(positive_period_info[0]).flatten(), pad_sample(positive_period_info[1]).flatten())
            )
        positive_sample = np.append(positive_sample, 1)
        np.savetxt(train, [positive_sample], delimiter=',', fmt='%u')

        # get negative_sample
        remove_list = [idx]
        negative_id = random.choice(remove_ids(remove_list, train_num))
        remove_list.append(negative_id)
        negative_period = periods_train[negative_id]
        negative_song_id = negative_period[0]
        negative_period_info = negative_period[1]
        while (negative_song_id == positive_song_id) or \
                (np.array_equal(positive_period_info[1], negative_period_info[1])):
            negative_id = random.choice(remove_ids(remove_list, train_num))
            negative_period = periods_train[negative_id]
            negative_song_id = negative_period[0]
            negative_period_info = negative_period[1]
            remove_list.append(negative_id)
        negative_sample = np.concatenate((pad_sample(positive_period_info[0]).flatten(),
                                          pad_sample(negative_period_info[1]).flatten())
                                         )
        negative_sample = np.append(negative_sample, 0)
        np.savetxt(train, [negative_sample], delimiter=',', fmt='%u')
    train.close()
    # get test 
    print("get test")
    test_num = len(periods_test)
    for idx, period in enumerate(periods_test):
        positive_song_id = period[0]
        positive_period_info = period[1]
        positive_sample = np.concatenate((pad_sample(positive_period_info[0]).flatten(),
                                          pad_sample(positive_period_info[1]).flatten())
                                         )
        positive_sample = np.append(positive_sample, 1)
        np.savetxt(test, [positive_sample], delimiter=',', fmt='%u')
        remove_list = [idx]
        for can_id in range(49):
            negative_id = random.choice(remove_ids(remove_list, test_num))
            remove_list.append(negative_id)
            negative_period = periods_test[negative_id]
            negative_song_id = negative_period[0]
            negative_period_info = negative_period[1]
            while (negative_song_id == positive_song_id) or \
                    (np.array_equal(positive_period_info[1], negative_period_info[1])):
                negative_id = random.choice(remove_ids(remove_list, test_num))
                negative_period = periods_test[negative_id]
                negative_song_id = negative_period[0]
                negative_period_info = negative_period[1]
                remove_list.append(negative_id)
            negative_sample = np.concatenate((pad_sample(positive_period_info[0]).flatten(),
                                              pad_sample(negative_period_info[1]).flatten())
                                             )
            negative_sample = np.append(negative_sample, 0)
            np.savetxt(test, [negative_sample], delimiter=',', fmt='%u')
    test.close()

