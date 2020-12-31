import pickle
import numpy as np
import json
from scipy.sparse import csc_matrix
import copy
import random

random.seed(0)




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
        time_info = meta_dict['time_signature_changes']
        # melody_id=meta_dict['melody_id']
        idx = 0
        tracks = []
        while str(idx) in meta_dict:
            matrix = reconstruct_sparse(
                loaded, 'track_{}'.format(idx))
            program = meta_dict[str(idx)]
            # tracks.append([program,ticks,matrix])
            tracks.append(matrix)
            idx += 1
        melody_info = reconstruct_sparse(loaded, 'melody_info')
        return time_info, tracks, melody_info





def get_bars(time_info, matrix):
    bars = []
    end_time = len(matrix[0])
    for time_signature in time_info[::-1]:
        signature = int(time_signature[0])
        time = int(time_signature[1])
        temp_matrix = matrix[:, time:end_time]
        bar_number = (end_time - time) // signature
        for i in reversed(range(bar_number)):
            start = time + i * signature
            end = start + signature
            bar_matrix = temp_matrix[:, start:end]
            bar_melody = bar_matrix[0:4, :]
            bar_count_info = len(np.nonzero(bar_melody)[0]) // 4
            if bar_count_info / signature >= 0.75:
                bars.insert(0, temp_matrix[:, start:end])
        end_time = time
    return bars



def split_sequence(name):
    file_list = pickle.load(open('/filenames/sequence/{}'.format(name), 'rb'))
    # split sequence into bars
    bar_path = '/dataset/bar/{}/'.format(name)

    for file in file_list:
        meta_dict = {}
        bar_info = {}
        time_info, tracks, melody_info = load_sequence(file)
        time_info_len = len(time_info)
        current_index = 1
        while current_index < time_info_len:
            if (time_info[current_index][1] - time_info[current_index - 1][1]) < time_info[current_index - 1][0]:
                time_info.remove(time_info[current_index - 1])
                time_info_len -= 1
            current_index += 1
        meta_dict['time_info'] = time_info
        song_matrix = copy.deepcopy(melody_info)
        for idx, track in enumerate(tracks):
            song_matrix = np.vstack((song_matrix, track))
        bars = get_bars(time_info, song_matrix)
        #  filter songs with no more than 8 bars
        if len(bars) >= 8:
            for idx, bar in enumerate(bars):
                # update_sparse(bar_info, bar, str(idx))
                bar_info[str(idx)] = bar



            filename = file.split('/')[-1]

            # save bars
            np.savez_compressed(bar_path + filename, **bar_info)
