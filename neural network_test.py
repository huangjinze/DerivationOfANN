from sympy import *
import numpy as np

x1 = Symbol('x1')
x2 = Symbol('x2')

X = np.array([x1, x2])
W = np.array([
    [0.2, 0.3, 9],
    [1, 2, 3]
])

W21 = np.array([
    [1, 2, 3]
])

b2 = np.array([0.001])

fa = np.dot(X, W)
active_func = 1 / (1-exp(-fa))
fb = W21 * active_func + b2
print(fb)
print(diff(fb, x1))