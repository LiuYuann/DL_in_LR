import os
import time
import numpy as np
from keras import layers
from keras import models
from keras.utils.np_utils import to_categorical
from iv1415 import lre15util

if (os.name == 'posix'):
    data_path = r'/home/lzc/lzc/ivc/r146_1_1/ivec15-lre/data'
    trial_key_file = r'/home/lzc/lzc/ivc/NIST_ivector15_keys/ivec15_lre_trial_key.v1.tsv'
elif (os.name == 'nt'):
    data_path = r'D:\LZC\ivc\r146_1_1\ivec15-lre\data'
    trial_key_file = r'D:\LZC\ivc\NIST_ivector15_keys\ivec15_lre_trial_key.v1.tsv'

dev_iv_h5file = os.path.join(data_path, 'ivec15_lre_dev_ivectors.h5')
train_iv_h5file = os.path.join(data_path, 'ivec15_lre_train_ivectors.h5')
test_iv_h5file = os.path.join(data_path, 'ivec15_lre_test_ivectors.h5')


def dnn():
    print("Data loading......")
    dev_ids, dev_durations, dev_languages, dev_ivec = lre15util.load_ivectors_h5(dev_iv_h5file)
    train_ids, train_durations, train_languages, train_ivec = lre15util.load_ivectors_h5(train_iv_h5file)
    test_ids, test_durations, test_languages, test_ivec = lre15util.load_ivectors_h5(test_iv_h5file)

    print("Preprocessing......")
    dev_ivec, train_ivec, test_ivec = lre15util.preprocess(dev_ivec, train_ivec, test_ivec)

    print("Model......")
    language_index, language_index_reverse = lre15util.get_language_index(train_languages)
    train_labels = [language_index_reverse.get(lang) for lang in train_languages]
    train_labels = to_categorical(train_labels)

    layer1 = layers.Dense(512, activation='relu', input_shape=(400,))
    layer2 = layers.Dense(512, activation='relu')
    layer3 = layers.Dense(50, activation='softmax')
    model = models.Sequential()
    model.add(layer1)
    model.add(layers.Dropout(0.4))
    model.add(layer2)
    model.add(layers.Dropout(0.4))
    model.add(layer3)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Training......")
    model.fit(x=train_ivec, y=train_labels, epochs=200, batch_size=1024, shuffle=True, verbose=2)

    print("Testing......")
    scores = model.predict(test_ivec)
    oos_scores = 0.7 * np.ones((scores.shape[0], 1))
    all_scores = np.concatenate([scores, oos_scores], axis=1 )

    all_max = all_scores.argmax(axis=1)
    answers = [language_index.get(i) for i in all_max]

    return answers


if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    test_id, language, trial_set = lre15util.read_trial_key(trial_key_file)
    answers = dnn()

    accuracy, accuracy_without_oos = lre15util.compute_accuracy(language, answers)
    print('Accuracy:{:.6f}    without oos:{:.6f}'.format(accuracy, accuracy_without_oos))
    cost, cost_without_oos = lre15util.compute_cost(language, answers)
    print('Cost:{:.6f}    without oos:{:.6f}'.format(cost, cost_without_oos))

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
