import os
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.stats
from sklearn.metrics import r2_score
from sympy import *

# 定义模型所在路径
weight_file_path = 'model/my_model.h5'

mm = MinMaxScaler()
ss = StandardScaler()

def print_keras_wegiths(weight_file_path):
    f = h5py.File(weight_file_path)  # 读取weights h5文件返回File类
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))  # 输出储存在File类中的attrs信息，一般是各层的名称

        for layer, g in f.items():  # 读取各层的名称以及包含层信息的Group类
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items(): # 输出储存在Group类中的attrs信息，一般是各层的weights和bias及他们的名称
                print("{}: {}".format(key, value))

            print("    Dataset:")
            for name, d in g.items(): # 读取各层储存具体信息的Dataset类
                print("{}: {}".format(name, d.value.shape)) # 输出储存在Dataset中的层名称和权重，也可以打印dataset的attrs，但是keras中是空的
    finally:
        f.close()

def draw_data(data):
    plt.scatter([_ for _ in range(data.shape[0])], data)
    plt.show()

def DerivativeExpression(input, weights):

    X = np.array([Symbol(_) for _ in input])

    W1 = weights['sigmoid'][0]
    b1 = weights['sigmoid'][1]
    print('sigmoid层的权重值为：')
    print(W1)
    print('sigmoid层的偏差为：')
    print(b1)

    W2 = weights['linear'][0]
    b2 = weights['linear'][1]
    print('linear层的权重值为：')
    print(W2)
    print('linear层的偏差为：')
    print(b2)

    fa = np.dot(X, W1) + b1
    active_func = 1 / (1 - exp(-fa))
    fb = W2 * active_func + b2
    print(diff(fb, X[0]))


# 屏蔽waring信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""----------加载mnist数据集-------------"""
InputIndex = [
    'Rn', 'PAR', 'fdif', 'PARdif', 'PARdir',
    'Ta', 'Ts', 'Vpd', 'RH', 'Ustar', 'O3',
    'VWC5', 'VWC25', 'VWC50', 'VWC100', 'VWC150', 'VWC200']
OutputIndex = ['NEE']
data = pd.read_csv('data/example.csv')

# 分割训练集合验证集，test_size=0.4代表从总的数据集合train中随机选取40%作为验证集，随机种子为0
train = data[InputIndex]
target = data[OutputIndex]

trX, teX, trY, teY = train_test_split(train, target, test_size=0.2, random_state=0)
X = ss.fit_transform(trX)
Y = ss.fit_transform(trY)
test_X = ss.transform(teX)
test_y = ss.transform(teY)


"""----------测试集原数据作图----------------"""
plt.figure(0)  # 创建图表1
plt.title('observe')
plt.scatter([_ for _ in range(test_y.shape[0])], test_y)



train = 1

if train == 0:
    """----------配置网络模型----------------"""
    # 配置网络结构
    model = Sequential()

    # 第一隐藏层的配置：输入17，输出20
    model.add(Dense(20, input_dim=17, activation='sigmoid'))
    # model.add(Dense(20, activation='sigmoid'))
    model.add(Dense(1))

    # 编译模型，指明代价函数和更新方法
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mae', optimizer=sgd, metrics=['mae'])

    """----------训练模型--------------------"""
    print("training starts.....")
    model.fit(X, Y, epochs=1000, verbose=1, batch_size=X.shape[0])

    """----------评估模型--------------------"""
    # 用测试集去评估模型的准确度
    cost = model.evaluate(test_X, test_y)
    print('\nTest accuracy:', cost)

    """----------模型存储--------------------"""
    save_model(model, weight_file_path)


else:
    model = load_model(weight_file_path)

    # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(test_y.T[0], model.predict(test_X).T[0])
    # print("R^2:", r_value**2)
    """----------计算R^2--------------------"""

    score = r2_score(test_y.T[0], model.predict(test_X).T[0])
    print(score)
    """----------预测作图--------------------"""
    plt.figure(1)  # 创建图表1
    plt.title('predict')
    plt.scatter([_ for _ in range(test_y.shape[0])], model.predict(test_X))
    # plt.show()

    """----------获取权重值--------------------"""
    weights = {}
    for layer in model.layers:
        weight = layer.get_weights()
        info = layer.get_config()
        # print(info['activation'])
        weights[info['activation']] = weight

    DerivativeExpression(InputIndex, weights)
