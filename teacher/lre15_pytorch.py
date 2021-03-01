import os
import time

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable

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

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out


def dnn():
    print("Data loading......")
    dev_ids, dev_durations, dev_languages, dev_ivec = lre15util.load_ivectors_h5(dev_iv_h5file)
    train_ids, train_durations, train_languages, train_ivec = lre15util.load_ivectors_h5(train_iv_h5file)
    test_ids, test_durations, test_languages, test_ivec = lre15util.load_ivectors_h5(test_iv_h5file)

    print("Preprocessing......")
    dev_ivec, train_ivec, test_ivec = lre15util.preprocess(dev_ivec, train_ivec, test_ivec)

    language_index, language_index_reverse = lre15util.get_language_index(train_languages)
    train_labels = [language_index_reverse.get(lang) for lang in train_languages]

    train_ivec = Variable(torch.FloatTensor(train_ivec))
    test_ivec = Variable(torch.FloatTensor(test_ivec))
    train_labels = Variable(torch.LongTensor(train_labels))

    # train_loader = torch.utils.data.DataLoader(dataset=train_ivec,
    #                                            batch_size=1000,
    #                                            shuffle=True)

    # Hyper Parameters   配置参数
    torch.manual_seed(1)  # 设置随机数种子，确保结果可重复
    input_size = 400
    hidden_size = 1000
    num_classes = 50
    num_epochs = 500  # 训练次数
    batch_size = 1000  # 批处理大小
    learning_rate = 0.001  # 学习率

    net = Net(input_size, hidden_size, num_classes)

    # Loss and Optimizer  定义loss和optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    px, py = [], []
    plt.rcParams['font.sans-serif'] = ['STSong']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # Train the Model   开始训练
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = net(train_ivec)
        loss = criterion(outputs, train_labels)
        loss.backward()  # 后向传播，计算梯度
        optimizer.step()  # 梯度更新

        # 打印并记录当前的index 和 loss
        print(epoch, " loss: ", loss.item())
        px.append(epoch)
        py.append(loss.item())

        # if epoch % 10 == 0:
        #     # 动态画出loss走向 结果：loss.png
        #     plt.cla()
        #     plt.title(u'训练过程的loss曲线')
        #     plt.xlabel(u'迭代次数')
        #     plt.ylabel('损失')
        #     plt.plot(px, py, 'r-', lw=1)
        #     plt.text(0, 0, 'Loss=%.4f' % loss.item(), fontdict={'size': 20, 'color': 'red'})
        #     plt.pause(0.1)
        # if epoch == num_epochs - 1:
        #     # 最后一个图像定格
        #     plt.show()

    scores = net(test_ivec)
    oos_scores = 0.6 * torch.ones((scores.shape[0], 1))
    all_scores = torch.concatenate([scores, oos_scores], axis=1)

    _, predicted = torch.max(all_scores.data, 1)  # 预测结果
    # total += labels.size(0)  # 正确结果
    # correct += (predicted == labels).sum()  # 正确结果总数
    #
    #
    #
    # print("Training......")
    # model.fit(x=train_ivec, y=train_labels, epochs=200, batch_size=1024, shuffle=True, verbose=0)
    #
    # print("Testing......")
    # scores = model.predict(test_ivec)
    #
    # all_max = scores.argmax(axis=1)
    answers = [language_index.get(i) for i in predicted.numpy()]

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
