import collections
import glob
import os
from os.path import basename

import mne
import numpy as np
import pandas as pd
import tqdm
from natsort import natsorted
from scipy import io
from scipy import signal
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    """high pass filter the input data
    Args:
        data (ndarray): data to be filtered
        cutoff (float): cut-off frequency
        fs (float): sampling frequency
        order (int, optional): order of the filter. Defaults to 5.
    Returns:
        ndarray: filtered data
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def process_ecg_mne(data, data_name, ch_names, ch_types, percent_overlap, sfreq):
    scaler = StandardScaler()

    # create mne info
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    info['description'] = data_name + 'dataset'

    # create mne raw object
    raw = mne.io.RawArray(data, info, verbose='error')

    # resample and high pass filter the data
    resamp_data = raw.copy().resample(sfreq=256, verbose='error')
    # resamp_data   = resamp_data.copy().filter(l_freq=config['filter_band'][0], h_freq=config['filter_band'][1], method='iir', picks='misc', verbose='error')

    # high pass filter the data
    filtered_data = butter_highpass_filter(resamp_data.get_data(), cutoff=0.5, fs=256, order=5)

    # normalize the data using z-score
    transformed_data = scaler.fit_transform(filtered_data.T).T
    # transformed_data1 = stats.zscore(resamp_data.get_data(), axis=1)

    # create mne raw object
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=256)
    transformed_mne = mne.io.RawArray(transformed_data, info, verbose='error')

    # segment the data into fixed length windows
    if percent_overlap == 0:
        epochs = mne.make_fixed_length_epochs(transformed_mne, duration=10,
                                              verbose='error')
    else:
        events = mne.make_fixed_length_events(transformed_mne,
                                              duration=10,
                                              overlap=percent_overlap * 10)
        epochs = mne.Epochs(transformed_mne,
                            events,
                            tmin=0,
                            tmax=10,
                            baseline=None,
                            verbose='error')

    return epochs


def process_labels_mne(data, data_name, ch_names, ch_types, sfreq):
    # create mne info
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    info['description'] = data_name + 'labels'

    # create mne raw object
    raw = mne.io.RawArray(data, info, verbose='error')

    # resample and high pass filter the data
    resamp_data = raw.copy().resample(sfreq=256, verbose='error')

    # segment the data into fixed length windows
    epochs = mne.make_fixed_length_epochs(resamp_data, duration=10, verbose='error')

    return epochs


def read_raw_dreamer_dataset(load_path):
    """read the DREAMER data from .mat file and save it into hdf5 format

    Args:
        load_path ([string]): path of the raw DREAMER dataset
        save_path ([string]): path to store the DREAMER dataset
        save ([bool]): boolean to save the data dictionary

    Returns:
        Data [dictionary]: dictionary loaded from the
    """
    Data = collections.defaultdict(dict)

    if os.path.exists(load_path):
        data = io.loadmat(load_path, simplify_cells=True)
        data_list = []
        ars_label_list = []
        val_label_list = []
        dom_label_list = []
        # extract the data of all 23 subjects
        for i in range(len(data['DREAMER']['Data'])):

            data_dic = collections.defaultdict(dict)

            # retrieve the ECG signals of all 18 videos for each subject
            for j in range(len(data['DREAMER']['Data'][i]['ECG']['stimuli'])):
                ecg_s = data['DREAMER']['Data'][i]['ECG']['stimuli'][j][:, 0].reshape(1, -1)
                # considering only one channel of ECG
                ecg_s = process_ecg_mne(ecg_s, 'dreamer', ['ECG1'], ['misc'], 0.5, 256).get_data()[:, :, :-1]

                val = data['DREAMER']['Data'][i]['ScoreValence'][j]
                ars = data['DREAMER']['Data'][i]['ScoreArousal'][j]
                dom = data['DREAMER']['Data'][i]['ScoreDominance'][j]

                fname = f"/data/0shared/hanyuhu/music/data/subject_{i}_video_{j}.npy"
                print(fname)
                np.save(fname, np.array(ecg_s))
                print(np.array(ecg_s).shape)
                val_label_list.append(val)
                ars_label_list.append(ars)
                dom_label_list.append(dom)
                data_list.append(fname)
        Data['ecg'] = data_list
        Data['val'] = val_label_list
        Data['ars'] = ars_label_list
        Data['dom'] = dom_label_list

        def feeling_map(s):
            """Refer: https://stackoverflow.com/questions/49586471/add-new-column-to-python-pandas-dataframe-based-on-multiple-conditions
            open to new ideas for this
            Feelings:
            "Happy", # Valence > 0 and Arousal >0
            "Calm",   # Valence > 0 and Arousal < 0
            "Angry",  # Valence <0 and Arousal > 0
            "Sad",  # Valence < 0 and Arousal < 0
            ]
            """
            if s['val'] >= 3 and s['ars'] >= 3:
                return 0
            elif s['val'] >= 3 and s['ars'] < 3:
                return 1
            elif s['val'] < 3 and s['ars'] >= 3:
                return 2
            elif s['val'] < 3 and s['ars'] < 3:
                return 3

        Data = pd.DataFrame(Data)
        Data['label'] = Data.apply(feeling_map, axis=1)
        Data.to_csv('ecg.csv')

    else:
        print("Please check the path")

    return Data


class DREAMER():
    def __init__(self, path='ecg.csv'):
        self.dataset = pd.read_csv(path)
        self.data = None
        self.valence = []
        self.arousal = []
        for (fname, val, ars) in zip(self.dataset['ecg'], self.dataset['val'], self.dataset['ars']):
            temp = np.load(fname).astype('float32')
            if self.data is None:
                self.data = temp
            else:
                self.data = np.concatenate([self.data, temp], axis=0)

            self.valence.extend([ars] * temp.shape[0])
            self.arousal.extend([val] * temp.shape[0])
        trans = MinMaxScaler(feature_range=(0, 1))
        self.valence = np.array(self.valence).reshape((-1, 1)).astype('float32')
        self.arousal = np.array(self.arousal).reshape((-1, 1)).astype('float32')
        self.valence = trans.fit_transform(self.valence)
        self.arousal = trans.fit_transform(self.arousal)

        self.label = np.concatenate([self.valence, self.arousal], axis=1)

        self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.data, self.label, test_size=0.2,
                                                                            shuffle=True)


class DREAMERDataset(Dataset):

    def __init__(self, data, encoded):
        self.data = data
        self.encoded = encoded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index].astype('float32')
        label = self.encoded[index].astype('float32')

        return data, label


def get_feelings(filename):
    """
    Parses the file and gets the mapping of songid and feeling
    returns: Pandas DataFrame
    """
    df = pd.read_csv(filename)
    # Remove unneccesay columns
    col_info = list(enumerate(df.columns))

    valence_columns_to_remove = [x for x in col_info if "valence" in x[1] and x[1] != " valence_mean"]
    df.drop(columns=[x[1] for x in valence_columns_to_remove], inplace=True)  # Valence related columns

    arousal_columns_to_remove = [x for x in col_info if "arousal" in x[1] and x[1] != " arousal_mean"]
    df.drop(columns=[x[1] for x in arousal_columns_to_remove], inplace=True)  # Arousal related columns

    # Rename columns
    df.rename(columns={' valence_mean': 'valence'}, inplace=True)
    df.rename(columns={' arousal_mean': 'arousal'}, inplace=True)

    # # pdb.set_trace()
    #
    # def feeling_map(s):
    #     """Refer: https://stackoverflow.com/questions/49586471/add-new-column-to-python-pandas-dataframe-based-on-multiple-conditions
    #     open to new ideas for this
    #     Feelings:
    #     "Happy", # Valence > 0 and Arousal >0
    #     "Calm",   # Valence > 0 and Arousal < 0
    #     "Angry",  # Valence <0 and Arousal > 0
    #     "Sad",  # Valence < 0 and Arousal < 0
    #     ]
    #     """
    #     if s['valence'] >= 5 and s['arousal'] >= 5:
    #         return 0
    #     elif s['valence'] >= 5 and s['arousal'] < 5:
    #         return 1
    #     elif s['valence'] < 5 and s['arousal'] >= 5:
    #         return 2
    #     elif s['valence'] < 5 and s['arousal'] < 5:
    #         return 3
    #
    # # Add a new column
    # df["Feeling"] = df.apply(feeling_map, axis=1)

    return df


def load_deam_dataset(path='/data/0shared/hanyuhu/samples/'):
    count = 0
    f = open(path + '1.txt', 'w')
    files_feelings = path + "annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
    df_1 = get_feelings(files_feelings)

    X_list = []
    file_idnex = [1474,1521,1527,1623,1645,1663,1668,1682,1724,1734]
    valence_list = []
    arousal_list = []
    for file in tqdm.tqdm(natsorted(glob.glob(path + "features/*.csv"))):
        if int(basename(file)[:-4]) in df_1['song_id']:
            f.write(file + '\n')
            df_features = pd.read_csv(file, sep=';')
            df_features_noframe = list(df_features.iloc[:88, 1:].values)
            if len(df_features_noframe) != 88:
                print(file)

            X_list.append(df_features_noframe)

            # y_list.extend([df_1[df_1['song_id'] == int(basename(file)[:-4])]['Feeling']])
            valence_list.extend([df_1[df_1['song_id'] == int(basename(file)[:-4])]['valence']])
            arousal_list.extend([df_1[df_1['song_id'] == int(basename(file)[:-4])]['arousal']])
            if int(basename(file)[:-4]) in file_idnex:
                print(int(basename(file)[:-4]), count)
            count += 1

    print(np.array(X_list).shape, np.array(valence_list).shape)

    X_scaler = StandardScaler()
    X_list = X_scaler.fit_transform(np.vstack(np.array(X_list)).flatten()[:, np.newaxis].astype(float))
    X_list = X_list.reshape((-1, 88, 260))

    np.save('/data/0shared/hanyuhu/samples/deam_npy/X', np.array(X_list))
    np.save('/data/0shared/hanyuhu/samples/deam_npy/valence', np.array(valence_list))
    np.save('/data/0shared/hanyuhu/samples/deam_npy/arousal', np.array(arousal_list))
    f.close()


class DEAM():
    def __init__(self, path='/data/0shared/hanyuhu/samples/deam_npy/'):
        self.data = np.load(path + 'X.npy').astype('float32')
        self.valence = np.load(path + 'valence.npy').astype('float32')
        self.arousal = np.load(path + 'arousal.npy').astype('float32')

        trans = MinMaxScaler(feature_range=(0, 1))
        self.valence = np.array(self.valence).reshape((-1, 1))
        self.valence = trans.fit_transform(self.valence)
        self.arousal = np.array(self.arousal).reshape((-1, 1))
        self.arousal = trans.fit_transform(self.arousal)

        self.label = np.concatenate([self.valence, self.arousal], 1)

        self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.data, self.label, test_size=0.2,
                                                                            shuffle=False)


class DEAMDataset(Dataset):

    def __init__(self, data, encoded):
        self.data = data
        self.encoded = encoded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index].astype('float32')
        label = self.encoded[index].astype('float32')

        # data = torch.from_numpy(data).float()

        return data, label


class MyDataset(Dataset):
    def __init__(self, ecg_data, ecg_label, music_data, music_label, mean = 0.46829593):
        self.ecg_data = ecg_data
        self.ecg_label = ecg_label
        self.music_data = music_data
        self.music_label = music_label
        self.mean = mean
        self.distances = pairwise_distances(ecg_label.reshape(-1, 2), music_label.reshape((-1, 2)))
        self.similarity = np.exp(-self.distances / self.mean)


    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, item):
        ecg = self.ecg_data[item]
        ecg_label = self.ecg_label[item].reshape((2))

        # same_index = np.where(pairwise_distances(self.music_label, ecg_label) < 0.3)[0]
        # print(len(same_index))

        music_index = np.random.randint(0, len(self.music_label))
        music = self.music_data[music_index]
        music_label = self.music_label[music_index].reshape((-1, 2))

        distance = self.similarity[item][music_index]

        return ecg, ecg_label.reshape((2)), music, music_label.reshape((2)), distance.reshape((1))


class MyTripletDataset(Dataset):
    def __init__(self, ecg_data, ecg_label, music_data, music_label, th=0.1):
        self.ecg_data = ecg_data
        self.ecg_label = ecg_label
        self.music_data = music_data
        self.music_label = music_label
        self.th = th
        self.distances = pairwise_distances(ecg_label.reshape(-1, 2), music_label.reshape((-1, 2)))
        self.mean = 0.46829593
        self.similarity = np.exp(-self.distances / self.mean)

    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, item):
        ecg_index = item
        music_index = np.random.choice(len(self.music_data))

        # another_index = np.random.choice(len(self.music_data))

        another_music_index = np.random.choice(len(self.music_data))
        another_ecg_index = np.random.choice(len(self.ecg_data))
        while another_music_index == music_index and another_ecg_index == ecg_index:
            another_music_index = np.random.choice(len(self.music_data))
            another_ecg_index = np.random.choice(len(self.ecg_data))

        ecg = self.ecg_data[ecg_index]
        ecg_label = self.ecg_label[ecg_index].reshape((-1, 2))
        music = self.music_data[music_index]
        music_label = self.music_label[music_index]

        an_ecg = self.ecg_data[another_ecg_index]
        an_ecg_label = self.ecg_label[another_ecg_index].reshape((-1, 2))
        an_music = self.music_data[another_music_index]
        an_music_label = self.music_label[another_music_index]

        return ecg, ecg_label.reshape((2)), music, music_label.reshape((2)), an_ecg, an_ecg_label.reshape((2)), an_music, an_music_label.reshape((2)), \
               self.similarity[ecg_index][music_index] + 0.00001, self.similarity[ecg_index][another_music_index] + 0.00001, \
               self.similarity[another_ecg_index][music_index] + 0.00001

