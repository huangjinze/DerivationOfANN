import algopy
from sympy import *
import sympy as sp
from algopy import UTPM
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras import optimizers
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.stats
from sklearn.metrics import r2_score
# from sympy import *

# 定义模型所在路径
weight_file_path = 'model/my_model.h5'

def save_weights(weights_dict):
    print(weights_dict)
    for key, value in weights_dict:
        print(key)
        # df = pd.DataFrame(value)
        # df.to_csv('weights/'+key+'.csv')

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

# input: 输入指标，为list形式
# weights：每层的权重，字典形式
# derivate_target: 求偏导目标的索引
# return DerivativeExpression
def DerivativeExpression1(sympy_X, weights, Target):
    assert type(sympy_X) == list
    assert type(weights) == dict
    assert type(Target) == Symbol

    X = np.array(sympy_X)
    W1 = weights['sigmoid'][0]
    b1 = weights['sigmoid'][1]
    # print('sigmoid层的权重值为：')
    # print(W1)
    # print('sigmoid层的偏差为：')
    # print(b1)

    W2 = weights['linear'][0]
    b2 = weights['linear'][1]
    # print('linear层的权重值为：')
    # print(W2)
    # print('linear层的偏差为：')
    # print(b2)

    fa = np.dot(X, W1) + b1
    active_func = np.array([1 / (1 - sp.exp(-_)) for _ in fa])
    fb = np.dot(active_func, W2) + b2
    # print("the derivation of ", input[derivate_target_index])
    # print(diff(fb, X[derivate_target_index]))
    return diff(fb[0], Target)

# 两层隐藏层
def DerivativeExpression_jac1(sympy_X, weights):
    assert type(weights) == dict

    X = sympy_X
    W1 = weights['dense_1'][0]
    b1 = weights['dense_1'][1]

    W2 = weights['dense_2'][0]
    b2 = weights['dense_2'][1]



    fa = np.dot(X, W1) + b1

    # active_fa = np.array([1 / (1 - algopy.exp(-_)) for _ in fa])
    active_fa = 1 / (1 + algopy.exp(-fa))
    fb = np.dot(active_fa, W2) + b2

    # print("the derivation of ", input[derivate_target_index])
    # print(diff(fb, X[derivate_target_index]))
    return fb[0]

# 使用jacobian矩阵计算
def DerivativeExpression_jac2(sympy_X, weights):
    assert type(weights) == dict

    X = sympy_X
    W1 = weights['dense_1'][0]
    b1 = weights['dense_1'][1]

    W2 = weights['dense_2'][0]
    b2 = weights['dense_2'][1]

    W3 = weights['dense_3'][0]
    b3 = weights['dense_3'][1]


    fa = np.dot(X, W1) + b1

    # active_fa = np.array([1 / (1 - algopy.exp(-_)) for _ in fa])
    active_fa = 1 / (1 + algopy.exp(-fa))
    fb = np.dot(active_fa, W2) + b2
    # active_fb = np.array([1 / (1 - algopy.exp(-_)) for _ in fb])
    active_fb = 1 / (1 + algopy.exp(-fb))
    fc = np.dot(active_fb, W3) + b3

    # print("the derivation of ", input[derivate_target_index])
    # print(diff(fb, X[derivate_target_index]))
    return fc[0]
# Expression: the derivative Expression of Function
# Target: the name of derivative target, type is "sympy"
# InputIndex: the index data of derivative target, type is "list"
# InputIndex: the data of derivative target, type is "np.ndarray"
# return the calculate value
def DerivativeValue(Expression, Target, sympy_X, InputData):
    assert type(sympy_X) == list
    assert type(InputData) == np.ndarray
    assert len(sympy_X) == InputData.shape[1]
    # assert type(Target) == Symbol


    DerValue = []
    for RawNo in range(InputData.shape[0]):
        TargetValue = list(InputData[RawNo])
        data = dict(zip(sympy_X, TargetValue))
        value = Expression.evalf(subs=data)
        print("raws value is :", value)
        DerValue.append(value)
    return DerValue

def get_r2_numpy_manual(x, y):
    zx = (x - np.mean(x)) / np.std(x, ddof=1)
    zy = (y - np.mean(y)) / np.std(y, ddof=1)
    r = np.sum(zx * zy) / (len(x) - 1)
    return r ** 2


def ANN_Model(X, Y, test_X, test_y):

    """----------测试集原数据作图----------------"""
    # plt.figure(0)  # 创建图表1
    # plt.title('observe')
    # plt.scatter([_ for _ in range(test_y.shape[0])], test_y)


    # 训练次数
    # epochs = input('输入训练批次:\n')

    # loss_func = input('loss函数('
    #                   'mae[mean_absolute_error]\n'
    #                   'mse[mean_squared_error]\n'
    #                   'msle[mean_squared_logarithmic_error]\n'
    #                   'squared_hinge[squared_hinge]\n'
    #                   'logcosh[logcosh]\n'
    #                   '):\n')
    loss_func = 'mse'
    """----------配置网络模型----------------"""
    # 配置网络结构
    model = Sequential()


    # hidden_units = input('隐藏层单元数量:\n')
    hidden_units = 20
    # 第一隐藏层的配置：输入17，输出20
    if layers_num == 1:
        model.add(Dense(hidden_units, input_dim=len(InputIndex), activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
    else:
        hidden_units1 = 20
        hidden_units2 = 16
        model.add(Dense(hidden_units1, input_dim=len(InputIndex), activation='sigmoid'))
        model.add(Dense(hidden_units2, activation='sigmoid'))
        model.add(Dense(1))


    # 编译模型，指明代价函数和更新方法
    Ada = optimizers.Adagrad(lr=0.018, epsilon=1e-06)
    model.compile(loss=loss_func, optimizer=Ada, metrics=[loss_func])


    """----------训练模型--------------------"""
    print("training starts.....")
    model.fit(X, Y, epochs=epochs, verbose=1, batch_size=256)

    """----------评估模型--------------------"""
    # 用测试集去评估模型的准确度
    cost = model.evaluate(test_X, test_y)
    print('\nTest accuracy:', cost)

    """----------模型存储--------------------"""
    save_model(model, weight_file_path)

    # 数据反归一化
    trueTestYv = org_teY
    temp = model.predict(test_X).reshape(-1, 1)
    predTestYv = (temp.T * npscale.reshape(-1, 1)[-1, :] + npminthred.reshape(-1, 1)[-1, :]).T

    save_data = {
        'Test': list(trueTestYv.T[0]),
        'Predict': list(predTestYv.T[0])
    }
    predict_predYv = pd.DataFrame(save_data)
    predict_predYv.to_csv('data/predict_test_value.csv')


    """----------计算R^2--------------------"""
    testYv = test_y.values.flatten()
    predYv = model.predict(test_X).flatten()
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(testYv, predYv)
    print('R square is: ', r_value ** 2)

    # 数据反归一化
    trueAllYv = org_target.values
    temp = model.predict(train).reshape(-1, 1)
    predAllYv = (temp.T * npscale.reshape(-1, 1)[-1, :] + npminthred.reshape(-1, 1)[-1, :]).T

    save_data = {
        'TrueData': list(trueAllYv.T[0]),
        'PredictData': list(predAllYv.T[0])
    }
    predict_AllYv = pd.DataFrame(save_data)
    predict_AllYv.to_csv('data/predict_all_value.csv')


    # 求偏导，然后直接作图

    # 获取每一层的权重
    weights = {}
    for layer in model.layers:
        weight = layer.get_weights()
        info = layer.get_config()
        weights[info['name']] = weight
        if info['name'] == 'dense_1':
            df_weights = pd.DataFrame(weight[0].T, columns=InputIndex)
        else:
            df_weights = pd.DataFrame(weight[0].T)

        df_bias = pd.DataFrame(weight[1].T, columns=['bias'])
        df = pd.concat([df_weights, df_bias], axis=1)
        df.to_csv('weights/' + info['name'] + '.csv')

    res = []
    for RawNo in range(train.shape[0]):
        TargetValue = list(train.loc[RawNo].values)
        x = UTPM.init_jacobian(TargetValue)
        # 求导函数，根据隐藏层数量选择
        # 1层：DerivativeExpression_jac1
        # 2层：DerivativeExpression_jac2
        if layers_num == 1:
            y = DerivativeExpression_jac1(x, weights)
        else:
            y = DerivativeExpression_jac2(x, weights)
        algopy_jacobian = UTPM.extract_jacobian(y)
        # 最后一列插入NEE
        res.append(list(algopy_jacobian))
    res = np.array(res)
    save_data = {
        'TrueNEE': list(trueAllYv.T[0]),
        'PredNEE': list(predAllYv.T[0])
    }
    predict_AllYv = pd.DataFrame(save_data)

    deriColumns = ['d'+str(_) for _ in train.columns.tolist()]
    result = pd.DataFrame(res, columns=deriColumns)
    result = pd.concat([result, original_data, predict_AllYv], axis=1)
    result.to_csv('data/result_jacobian.csv')

    result.dropna(inplace=True)


    for i in range(len(InputIndex)):
        plt.figure(i)  # 创建图表1
        IndexName = InputIndex[i]

        # result = result[(result['d'+IndexName] > -5000) & (result['d'+IndexName] < 5000)]

        y = abs(result['d'+IndexName].values * scale[IndexName]) / result.shape[0]
        x = result[IndexName].values
        plt.xlabel(IndexName)
        plt.ylabel("NEE-" + IndexName)
        plt.scatter(x, y, s=1)
        plt.savefig("res/" + IndexName + ".png")
    plt.show()

if __name__ == '__main__':

    # 屏蔽waring信息
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    """----------加载输入数据集-------------"""
    InputIndex = [
        'PAR',  'Ta', 'Ts', 'Vpd', 'Ustar',
        'RH', 'Fdif', 'PARdif', 'PARdir',
        'VWC_5', 'VWC_25', 'VWC_50', 'VWC_100', 'VWC_150', 'VWC_200'
        # 'PAR', 'Fdif', 'PARdif', 'PARdir', 'Vpd',
        # 'VWC_5', 'VWC_50',  'VWC_200',
        # 'Ta', 'Ts', 'Ustar',
        # 'RH',
        # 'VWC_25', 'VWC_100', 'VWC_150',
    ]
    OutputIndex = ['NEE']

    layers_num = 2
    epochs = 5000

    # NEE 调整在矩阵的最后一列
    allIndex = InputIndex+OutputIndex
    data = pd.read_csv('data/data712pc.csv', usecols=allIndex)
    original_data = data.copy()

    # 每一列放大缩小的倍数
    scale = [np.max(data[_]) - np.min(data[_]) for _ in allIndex]
    npscale = np.array(scale)
    scale = dict(zip(allIndex, scale))

    minthred = [np.min(data[_]) for _ in allIndex]
    npminthred = np.array(minthred)
    minthred = dict(zip(allIndex, minthred))

    scale_data = pd.DataFrame()

    for i in allIndex:
        scale_data[i] = (data[i]-minthred[i]) / scale[i]


    # 分割训练集合验证集，test_size=0.4代表从总的数据集合train中随机选取40%作为验证集，随机种子为0
    train = scale_data[InputIndex]
    target = scale_data[OutputIndex]

    org_train = original_data[InputIndex]
    org_target = original_data[OutputIndex]


    trX, teX, trY, teY = train_test_split(train, target, test_size=0.25)
    True_org_trX, True_org_teX, True_org_trY, True_org_teY = train_test_split(org_train, org_target, test_size=0.25, random_state=123)

    # 原始数据
    org_trX = (trX.values.T * npscale.reshape(-1, 1)[:len(InputIndex), :] + npminthred.reshape(-1, 1)[:len(InputIndex), :]).T
    org_teX = (teX.values.T * npscale.reshape(-1, 1)[:len(InputIndex), :] + npminthred.reshape(-1, 1)[:len(InputIndex), :]).T
    org_trY = (trY.values.T * npscale.reshape(-1, 1)[-1, :] + npminthred.reshape(-1, 1)[-1, :]).T
    org_teY = (teY.values.T * npscale.reshape(-1, 1)[-1, :] + npminthred.reshape(-1, 1)[-1, :]).T

    ANN_Model(trX, trY, teX, teY)

