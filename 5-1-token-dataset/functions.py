import pickle
import numpy as np
import json
import pandas as pd
from scipy.sparse import csc_matrix
import copy
import random

pitch_number = 655
octave_number = 11
velocity_number = 10
state_number = 3


def reconstruct_sparse(target_dict, name):
    """Return a reconstructed instance of `scipy.sparse.csc_matrix`."""
    return csc_matrix((target_dict[name + '_csc_data'],
                       target_dict[name + '_csc_indices'],
                       target_dict[name + '_csc_indptr']),
                      shape=target_dict[name + '_csc_shape']).toarray()


def load_sequence(file):
    with np.load(file) as loaded:
        if 'meta.json' not in loaded:
            raise ValueError("Cannot find 'info.json' in the npz file.")
        meta_dict = json.loads(loaded['meta.json'].decode('utf-8'))
        idx = 0
        tracks = []
        while str(idx) in meta_dict:
            matrix = reconstruct_sparse(
                loaded, 'track_{}'.format(idx))
            tracks.append(matrix)
            idx += 1
        melody_info = reconstruct_sparse(loaded, 'melody_info')
        return tracks, melody_info


def get_noset_notes(matrix):
    notes = []

    for idx, item in enumerate(matrix[3]):
        if matrix[0][idx] < 0:
            print('bad')
        if item == 3 and matrix[0][idx] > 0:
            notes.append(np.hstack((idx, matrix[:, idx])))
    return notes

def get_token_dataset(name):
    file_list = pickle.load(open('/filenames/sequence/{}'.format(name), 'rb'))


    melody_train_path = open('/dataset/token_dataset/melody_train_{}'.format(name), 'ab')
    melody_test_path = open('/dataset/token_dataset/melody_test_{}'.format(name), 'ab')

    harmony_train_path = open('/dataset/token_dataset/harmony_train_{}'.format(name), 'ab')
    harmony_test_path = open('/dataset/token_dataset/harmony_test_{}'.format(name), 'ab')


    label_train_path = open('/dataset/token_dataset/label_train_{}'.format(name), 'ab')
    label_test_path = open('/dataset/token_dataset/label_test_{}'.format(name), 'ab')

    for file in file_list:

        tracks, melody_info = load_sequence(file)
        song_matrix = copy.deepcopy(melody_info)

        for idx, track in enumerate(tracks):
            song_matrix = np.vstack((song_matrix, track))

        track_id = 0
        center_notes, melody_notes, harmony_context = get_context(track_id,
                                                                  melody_info,
                                                                  song_matrix)
        if center_notes == []:
            continue
        center_notes_set = copy.deepcopy(center_notes)
        melody_notes_set = copy.deepcopy(melody_notes)
        harmony_context_set = copy.deepcopy(harmony_context)

        for idx, track in enumerate(tracks):
            track_id += 1
            center_notes, melody_notes, harmony_context = get_context(track_id,
                                                                      track,
                                                                      song_matrix)
            if center_notes != []:
                center_notes_set = np.vstack((center_notes_set, center_notes))
                melody_notes_set = np.vstack((melody_notes_set, melody_notes))
                harmony_context_set = np.vstack((harmony_context_set, harmony_context))
        if center_notes_set != []:
            sample_set, label_set = generate_samples_labels(center_notes_set)
            melody_notes_set = melody_notes_set.repeat(5, axis=0)
            melody_notes_dataset = np.insert(melody_notes_set, 10, values=sample_set.T, axis=1)
            event_num=melody_notes_dataset.shape[0]
            train_num=int(event_num*0.9)
            melody_train = melody_notes_dataset[:train_num, :]
            melody_test = melody_notes_dataset[train_num:,:]
            np.savetxt(melody_train_path, melody_train, fmt='%u', delimiter=',')
            np.savetxt(melody_test_path, melody_test, fmt='%u', delimiter=',')
            #
            harmony_context_set = harmony_context_set.repeat(5, axis=0)
            harmony_context_dataset = np.insert(harmony_context_set, 0, values=sample_set[:, 1:].T, axis=1)

            harmony_train = harmony_context_dataset[:train_num, :]
            harmony_test = harmony_context_dataset[train_num:, :]
            np.savetxt(harmony_train_path , harmony_train, fmt='%u', delimiter=',')
            np.savetxt(harmony_test_path , harmony_test, fmt='%u', delimiter=',')
            #
            label_train = label_set[:train_num, :]
            label_test = label_set[train_num:, :]
            np.savetxt(label_train_path,label_train, fmt='%u', delimiter=',')
            np.savetxt(label_test_path, label_test, fmt='%u', delimiter=',')
    melody_train_path.close()
    melody_test_path.close()
    harmony_train_path.close()
    harmony_test_path.close()
    label_train_path.close()
    label_test_path.close()


def generate_samples_labels(center_notes_set):
    samples_set = []
    labels_set = []
    # if len(center_notes_set.shape)<2:
    #     center_notes_set=np.expand_dims(center_notes_set,axis=0)
    for idx, note in enumerate(center_notes_set):
        y_0 = np.array([1, 1, 1, 1])
        x_0 = note
        x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4 = generate_negative_samples(x_0)
        samples = np.vstack((x_0, x_1, x_2, x_3, x_4))
        labels = np.vstack((y_0, y_1, y_2, y_3, y_4))
        if idx == 0:
            samples_set = copy.deepcopy(samples)
            labels_set = copy.deepcopy(labels)
        else:
            samples_set = np.vstack((samples_set, samples))
            labels_set = np.vstack((labels_set, labels))
    return samples_set, labels_set


def get_context(track_id, matrix, song_matrix):
    melody_notes = []
    window = 2
    center_notes = []
    notes = get_noset_notes(matrix)
    harmony_context = []
    if len(notes) >= 2 * window + 2:
        for i in range(window, len(notes) - window):

            window_notes = []
            for j in reversed(range(window)):
                interval = notes[i - j - 1][0] - notes[i][0]
                window_notes = np.hstack((window_notes, np.hstack((interval, notes[i - j - 1][1:]))))
            for j in range(window):
                interval = notes[i + j + 1][0] - notes[i][0]
                window_notes = np.hstack((window_notes, np.hstack((interval, notes[i + j + 1][1:]))))
            if i == window:
                center_notes = copy.deepcopy(notes[i])
                melody_notes = copy.deepcopy(window_notes)
                harmony_context = copy.deepcopy(get_note_harmony(track_id, notes[i], song_matrix))
            else:
                center_notes = np.vstack((center_notes, notes[i]))
                melody_notes = np.vstack((melody_notes, window_notes))
                harmony_context = np.vstack((harmony_context, get_note_harmony(track_id, notes[i], song_matrix)))
    return center_notes, melody_notes, harmony_context


def generate_negative_samples(positive):
    negative_1 = copy.deepcopy(positive)
    negative_1[1], negative_1[2], negative_1[3], negative_1[4], y_1 = generate_level_1_sample(positive)
    negative_2 = copy.deepcopy(positive)
    negative_2[1], negative_2[2], negative_2[3], negative_2[4], y_2 = generate_level_2_sample(positive)
    negative_3 = copy.deepcopy(positive)
    negative_3[1], negative_3[2], negative_3[3], negative_3[4], y_3 = generate_level_3_sample(positive)
    negative_4 = copy.deepcopy(positive)
    negative_4[1], negative_4[2], negative_4[3], negative_4[4], y_4 = generate_level_4_sample(positive)
    return negative_1, negative_2, negative_3, negative_4, y_1, y_2, y_3, y_4


def get_choice():
    id = random.randint(1, 10)
    # chroma:octave:velocity:state=4:2:2:2
    choice_list = [4, 6, 8, 10]
    for i, num in enumerate(choice_list):
        if (id - num) <= 0:
            return i


def get_choice_reverse():
    id = random.randint(1, 15)
    # chroma:octave:velocity:state=3:4:4:4
    choice_list = [3, 7, 11, 15]
    for i, num in enumerate(choice_list):
        if (id - num) <= 0:
            return i


def generate_level_1_sample(positive):
    y = np.array([1, 1, 1, 1])
    id = get_choice()
    y[id] = 0
    pitch, octave, velocity, state = replace([id], positive)
    return pitch, octave, velocity, state, y


def generate_level_2_sample(positive):
    # replace two attribute
    y = np.array([1, 1, 1, 1])
    id_1 = get_choice()
    y[id_1] = 0
    while (True):
        id_2 = get_choice()
        if id_2 != id_1:
            y[id_2] = 0
            break
    pitch, octave, velocity, state = replace([id_1, id_2], positive)
    return pitch, octave, velocity, state, y


def generate_level_3_sample(positive):
    # replace three attribute
    y = np.array([0, 0, 0, 0])
    id = get_choice_reverse()
    y[id] = 1
    ids = [0, 1, 2, 3]
    ids.remove(id)
    pitch, octave, velocity, state = replace(ids, positive)
    return pitch, octave, velocity, state, y


def generate_level_4_sample(positive):
    # replace four attribute
    ids = [0, 1, 2, 3]
    pitch, octave, velocity, state = replace(ids, positive)
    y = np.array([0, 0, 0, 0])
    return pitch, octave, velocity, state, y


def remove_current(current, end):
    if current < end:
        scale = list(range(1, end + 1))
        scale.remove(int(current))
        return scale
    else:
        scale = list(range(1, end))
        return scale


def replace(ids, positive):
    pitch = positive[1]
    octave = positive[2]
    velocity = positive[3]
    state = positive[4]

    for id in ids:
        if id == 0:
            if octave < 11:
                pitch = random.choice(remove_current(pitch, pitch_number))
            else:
                # there are 8 pitchs in group 11
                pitch = random.choice(list(range(1, 9)) + list(range(13, pitch_number + 1)))
                pass
        else:
            if id == 1:
                if pitch > 8 and pitch < 13:
                    octave = random.choice(remove_current(octave, octave_number - 1))
                else:
                    octave = random.choice(remove_current(octave, octave_number))
            else:
                if id == 2:
                    velocity = random.choice(remove_current(velocity, velocity_number))
                else:
                    state = random.choice([1, 2])
    return pitch, octave, velocity, state


def get_note_harmony(track_id, note, song_matrix):
    harmony_notes = []
    step = note[0]
    step_info = song_matrix[:, step]
    ids = list(np.nonzero(step_info)[0])
    for id in ids:
        if id % 4 == 0 and id // 4 != track_id:
            pitch = song_matrix[id][step]
            octave = song_matrix[id + 1][step]
            velocity = song_matrix[id + 2][step]
            state = song_matrix[id + 3][step]
            harmony_notes.append([pitch, octave, velocity, state])
    harmony_length = len(harmony_notes)
    harmony_number = min(harmony_length, 25)
    harmony = []
    for i in range(harmony_number):
        harmony = np.hstack((harmony, harmony_notes[i]))
    for i in range(25 - harmony_number):
        harmony = np.hstack((harmony, [0, 0, 0, 0]))
    return harmony



