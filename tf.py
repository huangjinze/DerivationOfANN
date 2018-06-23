import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

min_max_scaler = MinMaxScaler()
ss = StandardScaler()
np.random.seed(1)  # 使每次随机产生的数都相同


# 导入数据

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
X = min_max_scaler.fit_transform(trX)
Y = min_max_scaler.fit_transform(trY)
test_X = min_max_scaler.transform(teX)
test_y = min_max_scaler.transform(teY)


shape_X = X.shape  # X, 行17列
shape_Y = Y.shape  # Y，  1列
m = X.shape[1]  # 样本数

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s

# 定义神经网络结构
def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]  # 输入层神经元个数
    n_h = 20  # 隐藏层神经元个数
    n_y = Y.shape[0]  # 输出神经元个数

    return (n_x, n_h, n_y)


# 初始化模型参数
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# 前向传播
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    # Implement Forward Propagation to calculate A2 (probabilities)
    # np.dot 代表矩阵相乘
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = Z2

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


# 计算cost
def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    Returns:
    """

    cost = np.sqrt(((A2 - Y) ** 2).mean())
    assert (isinstance(cost, float))

    return cost


# 反向传播
def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]  # 样本数目

    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Retrieve also A1 and A2 from dictionary "cache".
    # A1,A2是每一层的输出结果
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Backward propagation: calculate dW1, db1, dW2, db2.
    # 输出层误差
    dZ2 = A2 - Y
    # 隐藏层到输出层权重求导数，最后一层是线性值
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1, 2)))
    # dZ1 = np.multiply(np.dot(W2.T, dZ2), (A1*(1-A1)))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2,
             "dZ1": dZ1}

    return grads


# 更新参数
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule given above
    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients
    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# 打包模型
# num_iterations: 训练次数，如果nn_model中未指定，默认为10000
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(3)

    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


# 预测函数
def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    # predictions = (A2 > 0.5)

    return A2, cache


# 训练
parameters = nn_model(X.T, Y.T, n_h=20, num_iterations=3000, print_cost=True)
# 预测
predictions, cache = predict(parameters, test_X.T)
print('RMSE: ', (np.sqrt(((predictions - test_y.T) ** 2).mean())))

# print(parameters)

grads = backward_propagation(parameters, cache, test_X.T, test_y.T)

# 开始计算偏导数
w2 = np.sum(parameters['W2'], axis=1)/parameters['W2'].shape[0]
a = np.dot(parameters['W1'], test_X.T)
dI = sigmoid(a) * (1-sigmoid(a))


d = np.dot(parameters['W1'].T, dI) * w2
# print(d)
# 对计算结果进行归一化操作

draw_y = ss.fit_transform(d.T)
res = min_max_scaler.inverse_transform(test_X)

# 绘图
cloumns = train.columns.tolist()
for index in range(len(cloumns)):
    plt.figure(index)  # 创建图表1
    y = draw_y[:, index]
    x = res[:, index]
    plt.xlabel(cloumns[index])
    plt.ylabel("NEE-"+cloumns[index])
    plt.scatter(x, y)
    plt.savefig("res/"+cloumns[index]+".png")