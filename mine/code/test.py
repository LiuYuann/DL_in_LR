from keras.models import load_model
# from keras.utils.np_utils import to_categorical
import numpy as np


def count_Cost(data, Threshold=0.4):
    Poos = 0.23
    n = 50
    Cost = 0
    y_test = np.loadtxt('./test_label.csv', delimiter=',')
    index = np.unique(y_test).astype('int')
    y = y_test.tolist()
    label_count_index = {i: y.count(i) for i in index}
    label_correct_index = {i: 0 for i in index}
    for i in range(data.shape[0]):
        if np.max(data[i]) >= Threshold and np.argmax(data[i]) == y_test[i]:
            label_correct_index[y_test[i]] += 1
        elif np.max(data[i]) < Threshold and y_test[i] == 50:
            label_correct_index[50] += 1
    label_erro_index = {k: 1 - v / label_count_index[k] for k, v in label_correct_index.items()}
    for k, v in label_erro_index.items():
        if k != 50:
            Cost += v
    Cost = (((1 - Poos) / n) * Cost) + Poos * label_erro_index[50]
    return Cost


def count_no_threshold_Cost(data):
    count = 0
    y_test = np.loadtxt('../csv/test_label.csv', delimiter=',')
    for i in range(data.shape[0]):
        if np.argmax(data[i]) == y_test[i]:
            count += 1
    return count / 5000


if __name__ == '__main__':
    x_test = np.loadtxt('../csv/test_data.csv', delimiter=',')
    model = load_model('../my_model.h5')
    np.savetxt('./data.csv', model.predict(x_test))
    data = np.loadtxt('./data.csv')
    print(count_no_threshold_Cost(data))
