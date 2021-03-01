import os
import numpy as np
from iv1415 import lre15util

if (os.name == 'posix'):
    data_path = r'/home/lzc/lzc/ivc/r146_1_1/ivec15-lre/data'
elif (os.name == 'nt'):
    data_path = r'D:\LZC\ivc\r146_1_1\ivec15-lre\data'
    trial_key_file = r'D:\LZC\ivc\NIST_ivector15_keys\ivec15_lre_trial_key.v1.tsv'

dev_iv_tsv_file = os.path.join(data_path, 'ivec15_lre_dev_ivectors.tsv')
train_iv_tsv_file = os.path.join(data_path, 'ivec15_lre_train_ivectors.tsv')
test_iv_tsv_file = os.path.join(data_path, 'ivec15_lre_test_ivectors.tsv')

# dev_iv_file = os.path.join(data_path, 'ivec15_lre_dev_ivectors.npy')
# train_iv_file = os.path.join(data_path, 'ivec15_lre_train_ivectors.npy')
# test_iv_file = os.path.join(data_path, 'ivec15_lre_test_ivectors.npy')

dev_iv_h5file = os.path.join(data_path, 'ivec15_lre_dev_ivectors.h5')
train_iv_h5file = os.path.join(data_path, 'ivec15_lre_train_ivectors.h5')
test_iv_h5file = os.path.join(data_path, 'ivec15_lre_test_ivectors.h5')


def baseline_cos():
    # load ivector ids, durations , languages, and ivectors (as row vectors)
    print("\n 1. development data")
    dev_ids, dev_durations, dev_languages, dev_ivec = lre15util.load_ivectors_tsv(dev_iv_tsv_file)
    print("\n 2. training data")
    train_ids, train_durations, train_languages, train_ivec = lre15util.load_ivectors_tsv(train_iv_tsv_file)
    print("\n 3. test data")
    test_ids, test_durations, test_languages, test_ivec = lre15util.load_ivectors_tsv(test_iv_tsv_file)

    # compute the mean and whitening transformation over dev set only
    print("\n 4. mean and whiten")
    m = np.mean(dev_ivec, axis=0)
    S = np.cov(dev_ivec, rowvar=0)
    # print(S)
    D, V = np.linalg.eig(S)
    W = (1 / np.sqrt(D) * V).transpose().astype('float32')

    print("\n 5. center and whiten")
    # # center and whiten all i-vectors
    dev_ivec = np.dot(dev_ivec - m, W.transpose())
    train_ivec = np.dot(train_ivec - m, W.transpose())
    test_ivec = np.dot(test_ivec - m, W.transpose())

    print("\n 6. project all i_vectors into unit sphere")
    # # project all i-vectors into unit sphere
    print("\n   I-   >>6a. dev")
    dev_ivec /= np.sqrt(np.sum(dev_ivec ** 2, axis=1))[:, np.newaxis]
    print("\n   I-   >>6b. train")
    train_ivec /= np.sqrt(np.sum(train_ivec ** 2, axis=1))[:, np.newaxis]
    print("\n   I-   >>6c. test")
    test_ivec /= np.sqrt(np.sum(test_ivec ** 2, axis=1))[:, np.newaxis]

    # create a language model by taking mean over i-vectors
    # from that language (in this case each should have 300)
    avg_train_ivec = np.zeros((len(np.unique(train_languages)), train_ivec.shape[1]))
    avg_train_languages = []
    for i, language in enumerate(np.unique(train_languages)):
        avg_train_ivec[i] = np.mean(train_ivec[train_languages == language], axis=0)
        avg_train_languages.append(language)

    # project the avg train i-vectors into unit sphere
    avg_train_ivec /= np.sqrt(np.sum(avg_train_ivec ** 2, axis=1))[:, np.newaxis]

    print("\n 7. compute scores ")
    scores = np.dot(avg_train_ivec, test_ivec.transpose())

    print(" I-    LENGTH OF SCORES is ", len(scores))

    all_max = scores.argmax(axis=0)

    print(" I- length of answers : ", all_max.shape)

    answers = []
    for i in all_max:
        answers.append(avg_train_languages[i])

    return answers


def tsv2h5():
    lre15util.tsv2h5(dev_iv_tsv_file)
    lre15util.tsv2h5(train_iv_tsv_file)
    lre15util.tsv2h5(test_iv_tsv_file)

    dev_ids, dev_durations, dev_languages, dev_ivec = lre15util.load_ivectors_h5(dev_iv_h5file)
    train_ids, train_durations, train_languages, train_ivec = lre15util.load_ivectors_h5(train_iv_h5file)
    test_ids, test_durations, test_languages, test_ivec = lre15util.load_ivectors_h5(test_iv_h5file)


if __name__ == '__main__':
    # tsv2h5()

    test_id, language, trial_set = lre15util.read_trial_key(trial_key_file)
    answers = baseline_cos()
    accuracy, accuracy_without_oos = lre15util.compute_accuracy(language, answers)
    print('Accuracy:{:.6f}    without oos:{:.6f}'.format(accuracy, accuracy_without_oos))
    cost, cost_without_oos = lre15util.compute_cost(language, answers)
    print('Cost:{:.6f}    without oos:{:.6f}'.format(cost, cost_without_oos))
