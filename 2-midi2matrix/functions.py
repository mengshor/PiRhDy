import pickle
import numpy as np
import json
import zipfile
from scipy.sparse import csc_matrix, save_npz, load_npz
from statistics import mean
import copy
import random

random.seed(0)
import os
from itertools import combinations

# number without the rest note
pitch_number = 655
octave_number = 11
velocity_number = 10


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

def get_end_time(instrument_info):
    end_time = [note[3] for instrument in instrument_info for note in instrument[1]]
    if end_time == []:
        pass
    else:
        return int(max(end_time))


def get_matrix(instrument, end_time):
    note_matrix = np.zeros((128, end_time), dtype=int)
    for note in instrument:
        start = int(note[2]) - 1
        end = int(note[3]) - 1
        velocity = int(note[1])
        pitch = int(note[0]) - 1
        if pitch > 127:
            break
        # start=velocity+10; end=-velocity-10; if start==end, set as velocity+20
        if end != start:
            # onset=11, velocity num is from 1 to 10
            note_matrix[pitch, start] = velocity + 10
            # offset=-11
            note_matrix[pitch, end] = -(velocity + 10)
            note_matrix[pitch, start + 1:end] = velocity
        else:
            note_matrix[pitch, start] = velocity + 20
    return note_matrix


def get_empty(matrix):
    # remain track with more than 50% valid note events
    index = np.nonzero(matrix)
    total_length = matrix.shape[1]
    time_info = len(list(index[1]))
    if time_info / total_length > 0.5:
        return True
    else:
        return False


def get_pitch(matrix):
    # compute the average pitch of the track
    index = np.nonzero(matrix)
    pitch_info = list(index[0])
    pitch_mean = np.mean(pitch_info)
    return pitch_mean


def add_note_to_melody(matrix, note, step):
    velocity = matrix[note][step]
    # start=3,hold=2,end=1
    if velocity < 0:
        state = 1
    else:
        if velocity <= 10:
            state = 2
        else:
            state = 3
    # pitch, octave, velocity, state
    velocity = (abs(velocity) - 1) % 10 + 1
    # distinct chroma 0 (rest note) and octave 0 with element 0 by add 1
    return int(note % 12 + 1), int(note // 12 + 1), int(velocity), int(state)


def update_sparse(target_dict, sparse_matrix, name):
    """Turn `sparse_matrix` into a scipy.sparse.csc_matrix and update
    its component arrays to the `target_dict` with key as `name`
    suffixed with its component type string."""
    csc = csc_matrix(sparse_matrix)
    target_dict[name + '_csc_data'] = csc.data
    target_dict[name + '_csc_indices'] = csc.indices
    target_dict[name + '_csc_indptr'] = csc.indptr
    target_dict[name + '_csc_shape'] = csc.shape


def get_chord(matrix):
    from collections import defaultdict
    def list_duplicates(seq):
        tally = defaultdict(list)
        for i, item in enumerate(seq):
            tally[item].append(i)
        return ((key, locs) for key, locs in tally.items()
                if len(locs) > 1)

    index = np.nonzero(matrix)
    pitch_info = list(index[0])
    time_info = list(index[1])
    ticks = []
    chords = []
    for item in sorted(list_duplicates(time_info)):
        tick = int(item[0])
        index = item[1]
        chord = []
        for i in index:
            note = int(pitch_info[i])
            # not noteoff
            if matrix[note][tick] > 0:
                chord.append(note)
        if len(chord) >= 2:
            ticks.append(tick)
            chords.append(chord)
    return ticks, chords


def split_matrix(info_dict, meta_dict, matrix, new_idx, program):
    # split the track by octave ==> chord is limited in a octave
    for i in range(11):
        temp_matrix = matrix[0 + 12 * i:12 + 12 * i]
        if np.nonzero(temp_matrix)[0] != []:
            new_matrix = np.zeros((128, len(matrix[0])), dtype=int)
            new_matrix[0 + 12 * i:12 + 12 * i] = temp_matrix
            ticks, chords = get_chord(new_matrix)
            meta_dict[str(new_idx)] = program
            meta_dict['track_ticks_{}'.format(new_idx)] = ticks
            meta_dict['track_chords_{}'.format(new_idx)] = chords
            update_sparse(info_dict, new_matrix, 'track_{}'.format(new_idx))
            new_idx += 1
    i = i + 1
    temp_matrix = matrix[0 + 12 * i:]
    if np.nonzero(temp_matrix)[0] != []:
        new_matrix = np.zeros((128, len(matrix[0])), dtype=int)
        new_matrix[0 + 12 * i:] = temp_matrix
        ticks, chords = get_chord(new_matrix)
        meta_dict[str(new_idx)] = program
        meta_dict['track_ticks_{}'.format(new_idx)] = ticks
        meta_dict['track_chords_{}'.format(new_idx)] = chords

        update_sparse(info_dict, new_matrix, 'track_{}'.format(new_idx))
        new_idx += 1
    return new_idx


def get_time_signature_changes(meta_data):
    time_signature_changes = meta_data[0]
    time_info = []
    for time in time_signature_changes:
        numerator = time.numerator * (32 / time.denominator)
        time = time.time
        time_info.append([int(numerator), int(time)])
    return time_info

# transform midi file to matrix
def midi_to_matrix(name):
    # pickle file that contains all file names in lakh_normalized dir, generated by write_filenames function
    file_list = pickle.load(open('filenames/lakh_normalized/{}'.format(name), 'rb'))
    root_path = 'dataset/matrix/{}/'.format(name)
    file_id = 0
    for file in file_list:
        print(file_id)
        file_id += 1
        data = np.load(file, allow_pickle=True)
        meta_dict = {'time_signature_changes': get_time_signature_changes(data['meta_info'])}
        instrument_info = data['instrument_info']
        info_dict = {}
        melody_id = 10000
        pitch_candidate = 0
        temp_instrument_info = []
        end_time = get_end_time(instrument_info)
        track_num = 0
        for idx, insturment in enumerate(instrument_info):
            program = int(insturment[0])
            note_matrix = get_matrix(insturment[1], end_time=end_time)
            if get_empty(note_matrix):
                # melody track is the one with the highest average pitch
                pitch = get_pitch(note_matrix)
                if pitch_candidate < pitch:
                    pitch_candidate = pitch
                    melody_id = track_num
                temp_instrument_info.append([program, note_matrix])
                track_num += 1
            else:
                continue
        if melody_id == 10000:
            # a song have and only have one melody track
            continue
        else:
            new_idx = 0
            for idx, instrument in enumerate(temp_instrument_info):
                program = int(instrument[0])
                matrix = instrument[1]
                # generate melody info
                # if there are multiple notes at the same time
                # reserve the note with "on" state and the highest pitch
                # otherwise, reserve the highest pitch
                if idx == melody_id:
                    melody_info = np.zeros((4, len(matrix[0])), dtype=int)
                    time_len = matrix.shape[1]
                    for step in range(time_len):
                        note_info = list(np.nonzero(matrix[:, step])[0])
                        if note_info != []:
                            note_candidate = []
                            for note in note_info:
                                if matrix[note][step] > 10:
                                    note_candidate.append(note)
                            if note_candidate != []:
                                melody_note = max(note_candidate)
                            else:
                                melody_note = max(note_info)
                            # add melody_note into the final melody track
                            melody_info[0][step], melody_info[1][step], melody_info[2][step], melody_info[3][
                                step] = add_note_to_melody(matrix, melody_note, step)
                            # remove the melody_note from the original track
                            matrix[melody_note][step] = 0
                    update_sparse(info_dict, melody_info, 'melody_info')
                    # reconstruct the remaining notes as a new accompaniment track
                    new_idx = split_matrix(info_dict, meta_dict, matrix, new_idx, program)
                else:
                    new_idx = split_matrix(info_dict, meta_dict, matrix, new_idx, program)
            filename = root_path + file.split('/')[-1]

            np.savez_compressed(filename, **info_dict)
            compression = zipfile.ZIP_DEFLATED
            with zipfile.ZipFile(filename, 'a') as zip_file:
                zip_file.writestr('meta.json', json.dumps(meta_dict), compression)
            pass



