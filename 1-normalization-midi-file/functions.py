import pickle
from pretty_midi import PrettyMIDI, TimeSignature, KeySignature
import numpy as np
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

def bpm_to_qpm(beat_tempo, numerator, denominator):
    """Converts from quarter notes per minute to beats per minute.

    Parameters
    ----------
    quarter_note_tempo : float
        Quarter note tempo.
    numerator : int
        Numerator of time signature.
    denominator : int
        Denominator of time signature.

    Returns
    -------
    bpm : float
        Tempo in beats per minute.
    """

    if not (isinstance(beat_tempo, (int, float)) and
            beat_tempo > 0):
        raise ValueError(
            'Quarter notes per minute must be an int or float '
            'greater than 0, but xx was supplied')
    if not (isinstance(numerator, int) and numerator > 0):
        raise ValueError(
            'Time signature numerator must be an int greater than 0, but {} '
            'was supplied.'.format(numerator))
    if not (isinstance(denominator, int) and denominator > 0):
        raise ValueError(
            'Time signature denominator must be an int greater than 0, but {} '
            'was supplied.'.format(denominator))

    # denominator is whole note
    if denominator == 1:
        return beat_tempo * 4.0
    # denominator is half note
    elif denominator == 2:
        return beat_tempo * 2.0
    # denominator is quarter note
    elif denominator == 4:
        return beat_tempo
    # denominator is eighth, sixteenth or 32nd
    elif denominator in [8, 16, 32]:
        # simple triple
        if numerator == 3:
            return beat_tempo / 2.0
        # compound meter 6/8*n, 9/8*n, 12/8*n...
        elif numerator % 3 == 0:
            return beat_tempo * 3.0 / 2.0
        # strongly assume two eighths equal a beat
        else:
            return beat_tempo
    else:
        return beat_tempo


def velocity_num(number):
    # http://www.music-software-development.com/midi-tutorial.html
    velocity_list = [8, 20, 31, 42, 53, 64, 80, 96, 112, 127]
    for i, velocity in enumerate(velocity_list):
        if (number - (velocity + 1)) < 0:
            return i + 1


def get_pitch_class_histogram(notes, use_duration=True, use_velocity=True, normalize=True):
    weights = np.ones(len(notes))
    # Assumes that duration and velocity have equal weight
    if use_duration:
        weights *= [note.end - note.start for note in notes]
    if use_velocity:
        weights *= [note.velocity for note in notes]

    histogram, _ = np.histogram([n.pitch % 12 for n in notes],
                                bins=np.arange(13),
                                weights=weights,
                                density=normalize)
    if normalize:
        histogram /= (histogram.sum() + (histogram.sum() == 0))
    return histogram


def time_group(start, time_list):
    for time in time_list[::-1]:
        if start - time >= 0:
            return time_list.index(time)


def seconds_to_32th(time, tempo):
    # change seconds into 32th
    return time * tempo * 8


def time_to_beat(start, end, t_times, tempo_changes):
    # change time into beats
    t_group = time_group(start, t_times)
    tempo_value = tempo_changes[1][t_group]
    tempo = tempo_value / 60.0
    start_beat = seconds_to_32th(start, tempo)
    end_beat = seconds_to_32th(end, tempo)
    duration = end_beat - start_beat
    return start_beat, end_beat, duration


def normalization_midi_file(dirs):
    for name in dirs:
        # a file that contains all file names of the lakh dataset, using write_filenames function
        file_lists = pickle.load(open('filenames/lakh_dataset/{}'.format(name), 'rb'))

        # dir for normalized files
        root_path = 'dataset/lakh_normalized/{}/'.format(name)

        # allowed time signature, covering more than 90% of the dataset
        time_list = [4 / 4, 2 / 4, 3 / 4, 1 / 4, 1 / 8, 6 / 8, 3 / 8, 5 / 8]

        # krumhansl-schmuckler key-finding
        key_profile = pickle.load(open('key_profile.pickle', 'rb'))
        for file in file_lists:
            try:
                midi_data = PrettyMIDI(file)
                # time_signature changes
                ts_check = 1
                time_signature_changes = midi_data.time_signature_changes
                T_times = []
                if time_signature_changes == []:
                    time_signature_changes.append(TimeSignature(numerator=4, denominator=4, time=0.0))
                    T_times.append(0.0)
                else:
                    for time in time_signature_changes:
                        if not time.numerator / time.denominator in time_list:
                            ts_check = 0
                            break
                        else:
                            T_times.append(time.time)
                if ts_check:
                    # check resolution
                    resolution = midi_data.resolution
                    if resolution > 1024:
                        continue

                    # key_signature_changes
                    key_signature_changes = midi_data.key_signature_changes
                    K_times = []
                    if key_signature_changes == []:
                        key_signature_changes.append(KeySignature(key_number=0, time=0.0))
                    # get changing times
                    for key in key_signature_changes:
                        K_times.append(key.time)

                    # tempo_changes
                    tempo_changes = midi_data.get_tempo_changes()
                    t_times = []
                    if tempo_changes == []:
                        tempo_changes = [[0.0],
                                         bpm_to_qpm(midi_data.estimate_tempo(), time_signature_changes[0].numerator,
                                                    time_signature_changes[0].denominator)]
                        t_times.append(0.0)
                    else:
                        # get changing times
                        t_times = tempo_changes[0].tolist()

                    # estimate the real key signature changes
                    note_groups = [[] * len(K_times)]
                    for instrument in midi_data.instruments:
                        if not instrument.is_drum:
                            for note in instrument.notes:
                                if note.end > note.start:
                                    note_group = time_group(note.start, K_times)
                                    note_groups[note_group].append(note)
                    for i, notes in enumerate(note_groups):
                        histogram = get_pitch_class_histogram(notes)
                        key_candidate = np.dot(key_profile, histogram)
                        key_temp = np.where(key_candidate == max(key_candidate))
                        major_index = key_temp[0][0]
                        minor_index = key_temp[0][1]
                        major_count = histogram[major_index]
                        minor_count = histogram[minor_index % 12]
                        if major_count < minor_count:
                            key_signature_changes[i].key_number = minor_index
                        else:
                            key_signature_changes[i].key_number = major_index

                    instrument_info = []
                    for instrument in midi_data.instruments:
                        if instrument.is_drum:
                            continue
                        else:
                            note_info = []
                            program = instrument.program
                            for note in instrument.notes:
                                start = note.start
                                end = note.end
                                start, end, duration = time_to_beat(start, end, t_times, tempo_changes)

                                if duration >= 1:
                                    k_group = time_group(start, K_times)
                                    real_key = key_signature_changes[k_group].key_number

                                    # transposite to C major or A minor
                                    if real_key <= 11:
                                        trans = 0 - real_key
                                    else:
                                        trans = 21 - real_key
                                    pitch = note.pitch + trans + 1

                                    velocity = velocity_num(note.velocity)
                                    note_info.append([pitch, velocity, start, end])
                            if note_info != []:
                                instrument_info.append([program, note_info])
                    if instrument_info != []:
                        meta_info = [time_signature_changes, key_signature_changes, tempo_changes, resolution]
                        info = {'meta_info': meta_info, 'instrument_info': instrument_info}

                        file_name = file.split('/')[-1].split('.')
                        new_file = root_path + str(file_name[0]) + '.npz'
                        np.savez_compressed(new_file, **info)
            except Exception:
                pass