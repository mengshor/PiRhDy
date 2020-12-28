import pickle
import numpy as np
import json
import zipfile
from scipy.sparse import csc_matrix
from statistics import mean
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

def matrix_to_sequence(name):
    # a pickle file that contains all file names of the "matrix/name" dir, using write_filenames function
    file_list = pickle.load(open('filenames/matrix/{}'.format(name), 'rb'))
    for idx, file in enumerate(file_list):
        print(idx)
        rewrite_file(file)

def reconstruct_sparse(target_dict, name):
    """Return a reconstructed instance of `scipy.sparse.csc_matrix`."""
    return csc_matrix((target_dict[name + '_csc_data'],
                       target_dict[name + '_csc_indices'],
                       target_dict[name + '_csc_indptr']),
                      shape=target_dict[name + '_csc_shape']).toarray()

def update_sparse(target_dict, sparse_matrix, name):
    """Turn `sparse_matrix` into a scipy.sparse.csc_matrix and update
    its component arrays to the `target_dict` with key as `name`
    suffixed with its component type string."""
    csc = csc_matrix(sparse_matrix)
    target_dict[name + '_csc_data'] = csc.data
    target_dict[name + '_csc_indices'] = csc.indices
    target_dict[name + '_csc_indptr'] = csc.indptr
    target_dict[name + '_csc_shape'] = csc.shape

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
            # tracks.append([program,ticks,matrix])
            tracks.append([program, ticks, chords, matrix])
            idx += 1
        melody_info = reconstruct_sparse(loaded, 'melody_info')
        return time_info, tracks, melody_info

def get_mean_velocity(chord, step, matrix):
    # chord velocity is the average of all notes' velocities
    velocity = []
    state = 2
    for note in chord:
        if matrix[note][step] > 10:
            # -1 for real velocity number
            state = 3
            velocity.append(int((matrix[note][step] - 1) % 10 + 1))
        else:
            velocity.append(int(matrix[note][step]))

    return int(mean(velocity)), state

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
    # distinct chroma 0 and octave 0 with element 0 by add 1
    return int(note % 12 + 1), int(note // 12 + 1), int(velocity), int(state)

def rewrite_file(file):
    # rewrite matrix as quadruple (chroma, octave, velocity, state) sequence
    root_path = '/sequence/'
    chord_dict = np.load('../3-chord2chroma/chord_dict.npz', allow_pickle=True)['chord_dict']
    chord_index = np.load('../3-chord2chroma/chord_index.npz', allow_pickle=True)['chord_index']
    info_dict = {}
    time_info, tracks, melody_info = load_matrix(file)
    meta_dict = {'time_signature_changes': time_info}
    update_sparse(info_dict, melody_info, 'melody_info')
    for idx, track in enumerate(tracks):
        program = track[0]
        ticks = track[1]
        chords = track[2]
        matrix = track[3]
        track_sequence = np.zeros((4, len(matrix[0])), dtype=int)
        tick = 0
        for step in range(matrix.shape[1]):
            note_info = list(np.nonzero(matrix[:, step])[0])
            if tick < len(ticks) and step == ticks[tick]:
                octave = int(note_info[0] // 12) + 1
                chord = set(chords[tick])
                new_chord = []
                for note in chord:
                    new_chord.append(int(note % 12))
                new_chord = set(new_chord)
                # for note in ch
                velocity, state = get_mean_velocity(chord, step, matrix)
                if len(chord) > 5:
                    new_chord = {'over number'}
                # index = chord_dict.index(new_chord)
                index = np.where(chord_dict == new_chord)
                chord_id = chord_index[index[0]]

                track_sequence[0][step] = chord_id
                # octave id, +1 for 0,11-->1,12
                track_sequence[1][step] = octave
                # velocity
                track_sequence[2][step] = velocity
                track_sequence[3][step] = state
                tick += 1
            else:
                if note_info != []:
                    note_candidate = []
                    for note in note_info:
                        if matrix[note][step] > 10:
                            note_candidate.append(note)
                    if note_candidate != []:
                        melody_note = max(note_candidate)
                    else:
                        melody_note = max(note_info)
                    track_sequence[0][step], track_sequence[1][step], track_sequence[2][step], track_sequence[3][
                        step] = add_note_to_melody(matrix, melody_note, step)

        meta_dict[str(idx)] = program
        update_sparse(info_dict, track_sequence, 'track_{}'.format(idx))

    file_name = file.split('/')[-1]
    dir_name = file.split('/')[-2]
    filename = root_path + dir_name + '/' + file_name

    np.savez_compressed(filename, **info_dict)
    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(filename, 'a') as zip_file:
        zip_file.writestr('meta.json', json.dumps(meta_dict), compression)