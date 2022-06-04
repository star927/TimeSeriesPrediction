import numpy as np


def metric(pred, true):
    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))
    return mse, mae
