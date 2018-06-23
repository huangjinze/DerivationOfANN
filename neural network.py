import numpy as np

a = np.array([
    [1, 2, 1],
    [2, 3, 1]
])

b = np.array([
    [1, 2, 3],
    [2, 3, 4]
])

print(a*b)

print(np.sum(a, axis=0))