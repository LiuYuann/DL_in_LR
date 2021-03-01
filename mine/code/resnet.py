import numpy as np
from keras import layers, Input
from keras.models import Model
from keras.utils.np_utils import to_categorical
from test import *
from keras.utils import plot_model
hidden_num=4096
drop_out=0.5

def identity_block(X):
    X_shortcut = X

    X = layers.Dense(hidden_num)(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)
    X = layers.Dropout(drop_out)(X)

    X = layers.Dense(hidden_num)(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)
    X = layers.Dropout(drop_out)(X)

    X = layers.add([X, X_shortcut])
    X = layers.LeakyReLU()(X)

    return X


def build_model(input_dim, output_dim):
    m_input = Input(shape=[input_dim])

    X = layers.Dense(hidden_num)(m_input)
    X = identity_block(X)
    X = identity_block(X)

    X = layers.Dense(output_dim, activation='softmax')(X)

    model = Model(m_input, X)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='tt.png', show_shapes=True)
    return model


if __name__ == '__main__':
    y_train = to_categorical(np.loadtxt('../csv/train_label.csv', delimiter=',', dtype='int'))  # 将训练标签向量化
    y_test = to_categorical(np.loadtxt('../csv/test_label.csv', delimiter=',', dtype='int'))  # 将测试标签向量化
    x_train = np.loadtxt('../csv/train_data.csv', delimiter=',')  # 直接读取训练数据
    x_test = np.loadtxt('../csv/test_data.csv', delimiter=',')  # 直接读取测试数据
    model=build_model(400,50)
    model.fit(x=x_train, y=y_train, epochs=16, batch_size=256, shuffle=True)
    results = model.predict(x_test)
    rate = count_Correct_rate(results)
    print(rate)