
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


if __name__ == '__main__':

    """----------加载输入数据集-------------"""
    InputIndex = [
        # 'PAR', 'Ta', 'Ts', 'Vpd', 'Ustar',
        # 'RH', 'Fdif', 'PARdif', 'PARdir',
        # 'VWC_5', 'VWC_25', 'VWC_50', 'VWC_100', 'VWC_150', 'VWC_200'

        'PAR', 'Fdif', 'PARdif', 'PARdir', 'Vpd',
        'VWC_5', 'VWC_50', 'VWC_200'
    ]
    OutputIndex = ['NEE1']

    layers_num = 1
    epochs = 1000

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


    trX, teX, trY, teY = train_test_split(train, target, test_size=0.25, random_state=123)
    True_org_trX, True_org_teX, True_org_trY, True_org_teY = train_test_split(org_train, org_target, test_size=0.25, random_state=123)

    # 原始数据
    org_trX = (trX.values.T * npscale.reshape(-1, 1)[:len(InputIndex), :] + npminthred.reshape(-1, 1)[:len(InputIndex), :]).T
    org_teX = (teX.values.T * npscale.reshape(-1, 1)[:len(InputIndex), :] + npminthred.reshape(-1, 1)[:len(InputIndex), :]).T
    org_trY = (trY.values.T * npscale.reshape(-1, 1)[-1, :] + npminthred.reshape(-1, 1)[-1, :]).T
    org_teY = (teY.values.T * npscale.reshape(-1, 1)[-1, :] + npminthred.reshape(-1, 1)[-1, :]).T

    result = pd.read_csv('data/result_jacobian.csv')
    result.dropna(inplace=True)
    result.pop('Unnamed: 0')

    draw_x = result.values

    cloumns = result.columns.tolist()

    # for i in range(len(InputIndex)):
    #     plt.figure(i)  # 创建图表1
    #     IndexName = InputIndex[i]
    #
    #     # result = result[(result['d' + IndexName] > -5000) & (result['d' + IndexName] < 5000)]
    #
    #     yTrue = result['TrueNEE']
    #     yFake = result['PredNEE']
    #     x = result[IndexName].values
    #     plt.xlabel(IndexName)
    #     plt.ylabel("NEE-" + IndexName)
    #     plt.scatter(x, yTrue, s=1, c='r')
    #     plt.scatter(x, yFake, s=1, c='black')
    #     plt.legend(['observe', 'Predict'])
    #     # plt.savefig("res/" + IndexName + ".png")
    # plt.show()

    for i in range(len(InputIndex)):
        plt.figure(i)  # 创建图表1
        IndexName = InputIndex[i]

        # result = result[(result['d'+IndexName] > -500) & (result['d'+IndexName] < 500)]


        # y = result['d'+IndexName].values * scale[IndexName] / result.shape[0]
        y = (result['d'+IndexName].values.T * npscale.reshape(-1, 1)[i, :] + npminthred.reshape(-1, 1)[i, :]).T
        y = y / result.shape[0]
        x = result[IndexName].values
        plt.xlabel(IndexName)
        plt.ylabel("NEE-" + IndexName)
        plt.scatter(x, y, s=1)
        # plt.savefig("res/" + IndexName + ".png")
        plt.show()