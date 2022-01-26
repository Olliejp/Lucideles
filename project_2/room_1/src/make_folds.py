import pandas as pd
import numpy as np
import config

from sklearn import model_selection


def make_folds(
    df,
    target=None,
):
    df['kfold'] = -1

    # calculate the number of bins by Sturge's rule
    num_bins = int(np.floor(1 + np.log2(len(df))))

    # bin the target variables (20 bins)
    df.loc[:, 'bins'] = pd.cut(df[target], bins=num_bins, labels=False)

    # load kfold class from sklearn
    kf = model_selection.StratifiedKFold(n_splits=5)

    # modify the kfolds columns with the folds
    for fold, (_, valid_index) in enumerate(kf.split(X=df, y=df.bins.values)):
        df.loc[valid_index, 'kfold'] = fold

    df = df.drop('bins', axis=1)

    return df


if __name__ == "__main__":

    df_ill = pd.read_csv(config.TRAIN_ILL)
    df_ill_folds = make_folds(df_ill, target='ill')
    df_ill_folds.to_csv(config.DATA_PATH + "/train_ill_folds.csv", index=False)

    df_dgp = pd.read_csv(config.TRAIN_DGP)
    df_dgp_folds = make_folds(df_dgp, target='dgp')
    df_dgp_folds.to_csv(config.DATA_PATH + "/train_dgp_folds.csv", index=False)


