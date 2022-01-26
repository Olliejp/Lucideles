import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import config
import engine
from modules import drop_columns, calculate_mean_std, normalise, data_loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

EPOCHS = 500
BATCH_SIZE = 512
NUM_FOLDS = 5
LAYERS = 3
HIDDENSIZE=256
DROPOUT=0.0
TARGET='dgp'
DEVICE = 'cuda'


def run_training(
    fold,
    target,
    hidden_size,
    n_layers,
    dropout,
    save_model=False,
    plot=False
):

    if target == 'ill':
        df = pd.read_csv(config.TRAIN_ILL_WITH_FOLDS)
        df = drop_columns(df, config.DROP_COLUMNS)

    else:
        df = pd.read_csv(config.TRAIN_DGP_WITH_FOLDS)
        df = drop_columns(df, config.DROP_COLUMNS)

    # we have fold labels in the dataset, so for each fold
    # we subset these
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    # calculate the means and stds of all features + target
    # we only use train set stats so we dont introduce data
    # leakage in the xvalidation loop
    means, stds = calculate_mean_std(train_df)

    # we normalise specific columns and the target
    train_df = normalise(train_df,
                         means,
                         stds,
                         config.NORM_COLUMNS,
                         target)

    valid_df = normalise(valid_df,
                         means,
                         stds,
                         config.NORM_COLUMNS,
                         target)

    # subset the data on the training features and convert to
    # numpy arrays for train and valid sets for this fold
    xtrain = train_df[config.FEATURES].to_numpy(dtype='float64')
    ytrain = train_df[target].to_numpy(dtype='float64')
    ytrain = ytrain.reshape(len(ytrain), 1)

    xvalid = valid_df[config.FEATURES].to_numpy(dtype='float64')
    yvalid = valid_df[target].to_numpy(dtype='float64')
    yvalid = yvalid.reshape(len(yvalid), 1)

    # create dataloaders
    train_loader = data_loader(
        features=xtrain, targets=ytrain, batch_size=BATCH_SIZE)

    valid_loader = data_loader(
        features=xvalid, targets=yvalid, batch_size=BATCH_SIZE)

    model = engine.MLP(
        n_features=xtrain.shape[1],
        n_targets=ytrain.shape[1],
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout=dropout,
        batch_norm=True,
    )
    model.to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=20, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, threshold=1e-2, patience=1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, threshold=0.01, patience=2)
    eng = engine.Engine(
        model=model,
        optimizer=optimiser,
        scheduler=scheduler,
        device=DEVICE,
    )

    best_train_loss = np.inf
    best_valid_loss = np.inf
    early_stopping_iter = 20
    early_stopping_counter = 0

    epoch_number = []
    training_loss = []
    validation_loss = []

    tic = time.time()

    for epoch in range(EPOCHS):
        train_loss = eng.train(train_loader)
        valid_loss = eng.evaluate(valid_loader)

        epoch_number.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(valid_loss)

        if train_loss < best_train_loss:
            best_train_loss = train_loss

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if save_model:
                torch.save(model.state_dict(),
                           config.OUTPUTS_PATH + "/mlp_model_{target}.pth")
        else:
            early_stopping_counter += 1

        toc = time.time() - tic

        print(
            f"Fold: {fold}, epoch: {epoch}, training loss: {train_loss}, validation loss: {valid_loss}, best_validation_loss: {best_valid_loss}, time elapsed: {np.round(toc/60, 2)} minutes")

        if early_stopping_counter > early_stopping_iter:
            print(
                f"\ttraining stopped because validation loss did not improve after {early_stopping_iter} epochs")
            break

    if plot:
        plt.plot(epoch_number, training_loss, label="Training loss")
        plt.plot(epoch_number, validation_loss, label="Validation loss")
        plt.legend()
        plt.show()

    return best_train_loss, best_valid_loss

if __name__ == "__main__":

    fold_train_losses = []
    fold_valid_losses = []
    
    for fold in range(NUM_FOLDS):
        best_train_loss, best_val_loss = run_training(
            fold, TARGET, 
            HIDDENSIZE, LAYERS, 
            DROPOUT,)
    
        fold_train_losses.append(best_train_loss)
        fold_valid_losses.append(best_val_loss)
