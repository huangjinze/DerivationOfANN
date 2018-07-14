import pandas as pd
import numpy as np


def OCWRes(file1, file2):
    Dense1 = pd.read_csv(file1)
    Dense2 = pd.read_csv(file2)
    Dense1.pop('Unnamed: 0')
    Dense1.pop('bias')
    Dense2.pop('Unnamed: 0')
    Dense2.pop('bias')
    columns = Dense1.columns.tolist()
    Dense1Value = Dense1.values
    Dense2Value = Dense2.values
    OCW = Dense1Value * Dense2Value.T

    return np.mean(OCW, axis=0)

if __name__ == '__main__':
    file1 = 'weights/dense_1.csv'
    file2 = 'weights/dense_2.csv'
    result = OCWRes(file1, file2)
    print(result)
