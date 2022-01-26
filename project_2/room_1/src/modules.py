import pandas as pd
import numpy as np
import config
import torch
from torch.utils.data import TensorDataset, DataLoader


def drop_columns(
    df,
    drop_cols,
):
    return df[df.columns[~df.columns.isin(drop_cols)]]


def calculate_mean_std(
    df,
):
    return df.mean(axis=0), df.std(axis=0)


def normalise(
    df,
    mean,
    std,
    norm_cols,
    target,
    print_stats=False
):

    norm_cols.append(target)
    df.loc[:, norm_cols] = (df.loc[:, norm_cols] -
                            mean[norm_cols]) / std[norm_cols]

    if print_stats:
        for name in df.columns:
            print("%s mean: %.5f, %s std: %.2f" %
                  (name, df.loc[:, name].mean(), name, df.loc[:, name].std()))
        print('\n')

    return df


def reverse_targets(
    df,
    mean,
    std,
    target,
):

    return (df.loc[:, target] * std[target]) + mean[target]


def mse_loss(t1, t2):
    return torch.sum((t1-t2)**2) / t1.numel()


def mse(y, yhat):
    """
    Computes MSE - our models loss
    """
    return np.sum((np.array(y) - np.array(yhat))**2) / len(np.array(y))


def mae(y, yhat):
    """
    Computes Mean Absolute Error
    """
    return np.sum(np.abs(np.array(y) - np.array(yhat))) / len(np.array(y))


def accuracy_ill(y_hat, y):
    res = []
    for i in range(len(y)):
        if y[i] > 300:
            res.append(np.abs(y_hat[i] - y[i]) / y[i] < 0.2)
        elif 100 <= y[i] <= 300:
            res.append(np.abs(y_hat[i] - y[i]) / y[i] < 0.1)
        else:
            res.append(np.abs(y_hat[i] - y[i]) < 10)
    return np.mean(res)


def accuracy_dgp(y_hat, y):
    res = []
    for i in range(len(y)):
        if y[i] < 0.35 and y_hat[i] < 0.35: #Imperceptible glare
            res.append(True)
        elif 0.35 <= y[i] < 0.4 and 0.35 <= y_hat[i] < 0.4: #Perceptible glare
            res.append(True)
        elif 0.4 <= y[i] < 0.45 and 0.4 <= y_hat[i] < 0.45: #Disturbing glare
            res.append(True)
        elif y[i] >= 0.45 and y_hat[i] >= 0.45: #Intolerable glare
            res.append(True)
        else:
            res.append(False)
    return np.mean(res)


def data_loader(
    features,
    targets,
    batch_size
):

    X, Y = torch.from_numpy(features), torch.from_numpy(targets)
    X = X.type(torch.float64)
    Y = Y.type(torch.float64)
    torch_df = TensorDataset(X, Y)
    dloader = DataLoader(torch_df, batch_size, shuffle=False)

    return dloader
