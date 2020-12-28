import pickle
import numpy as np

def construct_phrase(name):
    file_list = pickle.load(open('/filenames/bars/{}'.format(name), 'rb'))
    root_path = '/dataset/phrase/'
    for file in file_list:
        phrase_info = {}
        bars_list = np.load(file, allow_pickle=True)
        idx = 0
        while str(idx) in bars_list:
            bar = bars_list[str(idx)]
            if idx % 4 == 0:
                bars = bar
            else:
                bars = np.hstack((bars, bar))
            if (idx + 1) % 4 == 0:
                phrase_info[str(idx // 4)] = bars
                # phrases.append(bars)
                bars = []
            idx += 1
        if bars != []:
            idx -= 1
            phrase_info[str(idx // 4)] = bars

        file_name = file.split('/')[-1]
        dir_name = file.split('/')[-2]
        filename = root_path + dir_name + '/' + file_name
        np.savez_compressed(filename, **phrase_info)
