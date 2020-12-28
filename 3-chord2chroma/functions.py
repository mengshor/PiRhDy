import pickle
import numpy as np
import json
from scipy.sparse import csc_matrix, save_npz, load_npz
import random

random.seed(0)
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

def reconstruct_sparse(target_dict, name):
    """Return a reconstructed instance of `scipy.sparse.csc_matrix`."""
    return csc_matrix((target_dict[name + '_csc_data'],
                       target_dict[name + '_csc_indices'],
                       target_dict[name + '_csc_indptr']),
                      shape=target_dict[name + '_csc_shape']).toarray()

def load_matrix(file):
    with np.load(file) as loaded:
        if 'meta.json' not in loaded:
            raise ValueError("Cannot find 'info.json' in the npz file.")
        meta_dict = json.loads(loaded['meta.json'].decode('utf-8'))
        time_info = meta_dict['time_signature_changes']
        # melody_id=meta_dict['melody_id']
        idx = 0
        tracks = []
        while str(idx) in meta_dict:
            matrix = reconstruct_sparse(
                loaded, 'track_{}'.format(idx))
            program = meta_dict[str(idx)]
            ticks = meta_dict['track_ticks_{}'.format(idx)]
            chords = meta_dict['track_chords_{}'.format(idx)]
            tracks.append([program, ticks, chords, matrix])
            idx += 1
        melody_info = reconstruct_sparse(loaded, 'melody_info')
        return time_info, tracks, melody_info


def collect_chords(name):
    # a pickle file that contains all file names of the "matrix/name" dir, using write_filenames function
    file_list = pickle.load(open('filenames/matrix/{}'.format(name), 'rb'))
    # a file contains all chords in this dir
    file_chord = 'chords/{}'.format(name)
    chord_dict = [['over number']]
    chord_count = [0]

    def add_dict(temp):
        if temp not in chord_dict:
            chord_dict.append(temp)
            chord_count.append(1)
        else:
            for index, item in enumerate(chord_dict):
                if item == temp:
                    chord_count[index] += 1

    file_id = 0
    for file in file_list:
        print(file_id)
        file_id += 1
        time_info, tracks, melody_info = load_matrix(file)
        for track in tracks:
            chords = track[2]
            for chord in chords:
                chord = set(chord)
                new_chord = []
                for note in chord:
                    new_chord.append(int(note % 12))
                if len(new_chord) > 5:
                    chord_count[0] += 1
                else:
                    add_dict(new_chord)
    chord_info = {'chord_dict': chord_dict, 'chord_count': chord_count}
    np.savez_compressed(file_chord, **chord_info)

def merge_chords():
    chord_dict = []
    chord_count = []
    names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    for name in names:
        data = np.load('chords/{}.npz'.format(name), allow_pickle=True)
        dict = data['chord_dict']
        count = data['chord_count']
        for idx, chord in enumerate(dict):
            chord = set(chord)
            check = 0
            for index, item in enumerate(chord_dict):

                if item == chord:
                    check = 1
                    chord_count[index] += count[idx]
                    break
            if not check:
                chord_dict.append(chord)
                chord_count.append(count[idx])
        pass
    chord_dict_info = {'chord_dict': chord_dict}
    chord_count_info = {'chord_count': chord_count}
    np.savez_compressed('chord_dict', **chord_dict_info)
    np.savez_compressed('chord_count', **chord_count_info)
    return chord_dict, chord_count

def get_chord_index():
    chord_dict, chord_count = merge_chords()
    chord_id = []
    idx = 13
    for i in range(len(chord_dict)):
        # frequency more than 1000
        if chord_count[i] > 1000:
            idx += 1
            chord_id.append(idx)

        else:
            # low frequency id=13
            chord_id.append(13)
    chord_index = {'chord_index': chord_id}
    np.savez_compressed('chord_index', **chord_index)
