from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
import numpy as np
from time import clock
from test import count_no_threshold_Cost


def fit(epochs):
    train_data = np.loadtxt('../csv/train_data.csv', delimiter=',')
    train_labels = to_categorical(np.loadtxt('../csv/train_label.csv', delimiter=','))
    layer1 = layers.Dense(512, activation='relu', input_shape=(400,))
    layer2 = layers.Dense(1024, activation='relu')  # 自动推导输入形状
    layer3 = layers.Dense(50, activation='softmax')
    model = models.Sequential()
    model.add(layer1)
    model.add(layers.Dropout(0.4))
    model.add(layer2)
    model.add(layers.Dropout(0.4))
    model.add(layer3)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(x=train_data[:10000], y=train_labels[:10000],validation_data=(train_data[10000:],train_labels[10000:]),epochs=500, batch_size=2048,shuffle=True)  # 训练模型
    model.fit(x=train_data, y=train_labels, epochs=epochs, batch_size=1024, shuffle=True, verbose=0)  # 训练模型
    # model.save('my_model.h5')
    return model


if __name__ == '__main__':
    # for i in range(48, 150):
    #     x_test = np.loadtxt('./test_data.csv', delimiter=',')
    #     model = fit(i)
    #     data = model.predict(x_test)
    #     print(i)
    #     print(count_no_threshold_Cost(data))
    x_test = np.loadtxt('../csv/test_data.csv', delimiter=',')
    model = fit(60)
    data = model.predict(x_test)
    print(count_no_threshold_Cost(data))
