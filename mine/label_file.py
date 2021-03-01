import numpy as np


def train():
    """
    训练数据规整函数
    :return: 语言编码字典
    """
    outer = np.loadtxt('./tsv/ivec15_lre_train_ivectors.tsv', delimiter='	', skiprows=1, dtype='str')
    label = np.unique(outer[:, 2])
    label_index = dict(zip(label, list(range(len(label)))))
    train_data = outer[:, 3:].astype('float32')
    train_label = []
    for i in outer:
        train_label.append(label_index[i[2]])
    train_label = np.asarray(train_label).reshape(15000, 1)
    np.savetxt('train_data.csv', train_data, fmt='%f', delimiter=',')  # 文本存储
    np.savetxt('train_label.csv', train_label, fmt='%f', delimiter=',')  # 文本存储
    return label_index


def test(label_index):
    """
    测试数据规整函数
    :param label_index: 语言编码字典
    :return: None
    """
    outer1 = np.loadtxt('./tsv/ivec15_lre_trial_key.v1.tsv', delimiter='	', skiprows=1, dtype='str', usecols=(0, 1))
    test_label = []
    label_index['out_of_set'] = 50
    for i in outer1:
        test_label.append(label_index[i[1]])
    test_label = np.asarray(test_label).reshape(6500, 1)
    np.savetxt('test_label.csv', test_label, fmt='%f', delimiter=',')  # 文本存储
    outer2 = np.loadtxt('./tsv/ivec15_lre_test_ivectors.tsv', delimiter='	', skiprows=1, dtype='str')
    test_data = outer2[:, 3:].astype('float32')
    np.savetxt('test_data.csv', test_data, fmt='%f', delimiter=',')  # 文本存储


def test_out():
    outer1 = np.loadtxt('./tsv/ivec15_lre_trial_key.v1.tsv', delimiter='	', skiprows=1, dtype='str', usecols=(0, 1))
    test_label = []
    sub = []
    for i in range(outer1.shape[0]):
        if outer1[i][1] != 'out_of_set':
            test_label.append(label_index[outer1[i][1]])
            sub.append(i)
    test_label = np.asarray(test_label).reshape(5000, 1)
    np.savetxt('test_no_outlabel.csv', test_label, fmt='%f', delimiter=',')  # 文本存储
    outer2 = np.loadtxt('./tsv/ivec15_lre_test_ivectors.tsv', delimiter='	', skiprows=1, dtype='str')
    test_data = []
    for i in sub:
        test_data.append(np.asarray(outer2[i][3:]).astype('float32'))
    test_data = np.asarray(test_data)
    np.savetxt('test_no_outdata.csv', test_data, fmt='%f', delimiter=',')  # 文本存储
    print(test_data.shape)


if __name__ == '__main__':
    label_index = train()
    print(label_index)
    # test(label_index)
