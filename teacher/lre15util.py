import csv
import h5py
import numpy as np
import pandas as pd


def load_ivectors_tsv(filename):
    ids = []
    durations = []
    languages = []
    ivectors = []
    with open(filename, 'rt') as infile:
        reader = csv.reader(infile, delimiter='\t')
        next(reader)

        for row in csv.reader(infile, delimiter='\t'):
            ids.append(row[0])
            durations.append(float(row[1]))
            languages.append(row[2])
            ivectors.append(np.asarray(row[3:], dtype=np.float32))

    return ids, np.array(durations, dtype=np.float32), np.array(languages), np.vstack(ivectors)


def load_ivectors_npy(filename):
    ids = []
    durations = []
    languages = []
    ivectors = []
    data = np.load(filename)
    for item in data:
        ids.append(item[0])
        durations.append(item[1])
        languages.append(item[2])
        iv = []
        for ind in range(3, 403):
            iv.append(item[ind])
        ivectors.append(iv)
    return ids, durations, languages, ivectors


def read_trial_key(key_file):
    trial_key = pd.read_csv(key_file, header=0, sep=r'\t', dtype=str, engine='python',
                            names=['test_id', 'language', 'trial_set'])
    test_id = trial_key['test_id']
    language = trial_key['language']
    trial_set = trial_key['trial_set']
    return test_id, language.values, trial_set


def compute_accuracy(key, answer):
    tf = key == answer
    accuracy = np.sum(tf) / len(tf)

    answer = np.asarray(answer)
    ind = key != 'out_of_set'
    key = key[ind]
    answer = answer[ind]

    tf = key == answer
    accuracy_without_oos = np.sum(tf) / len(tf)
    return accuracy, accuracy_without_oos


def compute_cost(key, answer):
    answer = np.asarray(answer)

    P_oos = 0.23
    n = 50

    ind_oos = key == 'out_of_set'
    key_oos = key[ind_oos]
    answer_oos = answer[ind_oos]
    P_error_oos = np.sum(key_oos != answer_oos) / len(key_oos)

    p_error = np.zeros(n, dtype=np.float32)
    unique_language = np.unique(key[~ind_oos])
    for i, language in enumerate(unique_language):
        ind_one = key == language
        key_one = key[ind_one]
        answer_one = answer[ind_one]
        p_error[i] = np.sum(key_one != answer_one) / len(key_one)

    return ((1 - P_oos) / n) * sum(p_error) + P_oos * P_error_oos, np.mean(p_error)


def preprocess(dev_ivec, train_ivec, test_ivec):
    m = np.mean(dev_ivec, axis=0)
    S = np.cov(dev_ivec, rowvar=0)
    # print(S)
    D, V = np.linalg.eig(S)
    W = (1 / np.sqrt(D) * V).transpose().astype('float32')

    # # center and whiten all i-vectors
    dev_ivec = np.dot(dev_ivec - m, W.transpose())
    train_ivec = np.dot(train_ivec - m, W.transpose())
    test_ivec = np.dot(test_ivec - m, W.transpose())

    dev_ivec /= np.sqrt(np.sum(dev_ivec ** 2, axis=1))[:, np.newaxis]
    train_ivec /= np.sqrt(np.sum(train_ivec ** 2, axis=1))[:, np.newaxis]
    test_ivec /= np.sqrt(np.sum(test_ivec ** 2, axis=1))[:, np.newaxis]

    return dev_ivec, train_ivec, test_ivec


def tsv2h5(tsv_filename, h5_filename=None):
    if not h5_filename:
        h5_filename = tsv_filename.replace('.tsv', '.h5', 1)

    ids, durations, languages, ivectors = load_ivectors_tsv(tsv_filename)
    ids = np.asarray(ids)
    languages = np.asarray(languages)

    file = h5py.File(h5_filename, 'w')
    ds_ids = file.create_dataset('ids', shape=ids.shape, dtype=h5py.special_dtype(vlen=str))
    ds_ids[:] = ids
    file.create_dataset('durations', data=durations)
    lang_ds = file.create_dataset('languages', shape=languages.shape, dtype=h5py.special_dtype(vlen=str))
    lang_ds[:] = languages
    file.create_dataset('ivectors', data=ivectors)
    file.close()

    return


def load_ivectors_h5(filename):
    file = h5py.File(filename, 'r')
    ids = file['ids'][:]
    durations = file['durations'][:]
    languages = file['languages'][:]
    ivectors = file['ivectors'][:]
    file.close()
    return ids, durations, languages, ivectors


def get_language_index(languages):
    unique_languages = np.unique(languages)
    language_index = dict([(key, value) for (key, value) in enumerate(unique_languages)])
    language_index[len(unique_languages)] = 'out_of_set'
    language_index_reverse = dict([(value, key) for (key, value) in language_index.items()])
    return language_index, language_index_reverse

# def create_ivec15_lre_numpy_binaries(tsv_infilepath, numpy_binary_outfilepath=None):
#     """
#     Purpose
#     -------
#     Transforms an IVEC15 LRE tab delimited data file
#     into a numpy binary (.npy)
#
#     Parameters
#     ----------
#     tsv_infilepath
#     numpy_binary_outfilepath
#     """
#     if not numpy_binary_outfilepath:
#         numpy_binary_outfilepath = \
#             tsv_infilepath.replace('.tsv', '.npy', 1)
#
#     # informs the np.genfromtxt method that field 0 and 2
#     # are string instead of the default 'float32'
#     converters_dict = {0: lambda s: str(s), 2: lambda s: str(s)}
#     test = np.genfromtxt(tsv_infilepath,
#                          converters=converters_dict,
#                          skip_header=1,
#                          delimiter="\t",
#                          dtype='float32')
#
#     np.save(numpy_binary_outfilepath, test)
