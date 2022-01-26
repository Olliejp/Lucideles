import config
from modules import drop_columns
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import time

import xgboost as xgb

NUM_FOLDS = 5
MAX_DEPTH_GRID = range(5, 15)
MIN_CHILD_WEIGHT_GRID = range(1, 4)
SUBSAMPLE_GRID = range(8, 11)
COLSAMPLE_GRID = range(8, 11)
ETA_GRID = np.linspace(0.01, 0.5, 20)

PARAMS = {
    'max_depth': 6,
    'min_child_weight': 1,
    'eta': 0.5,
    'subsample': 1,
    'colsample_bytree': 1,
    'objective': 'reg:squarederror',
    'eval_metric': 'mae'
}


class HYPER_XG:
    def __init__(self, target, params=PARAMS):
        self.params = params
        self.target = target

    def initiate_model(self):

        if self.target == 'ill':
            df = pd.read_csv(config.TRAIN_ILL_WITH_FOLDS)
            df = drop_columns(df, config.DROP_COLUMNS)

        else:
            df = pd.read_csv(config.TRAIN_DGP_WITH_FOLDS)
            df = drop_columns(df, config.DROP_COLUMNS)

        # Dataset is too large to run grid search (even with multithread on 16 cores),
        # taking just a subset to run hyperparam training
        df = df[df.kfold == 0].reset_index(drop=True)
        #valid_df = df[df.kfold == fold].reset_index(drop=True)

        # convert to XGB dataframes
        self.xtrain = xgb.DMatrix(df[config.FEATURES], label=df[self.target])
        #ytrain = xgb.DMatrix(df[config.FEATURES], label=df[target])

        model = xgb.cv(
            self.params,
            self.xtrain,
            num_boost_round=1000,
            seed=42,
            nfold=2,
            metrics={'mae'},
            early_stopping_rounds=10
        )

        self.best_mae = model['test-mae-mean'].min()
        print(self.best_mae)

    def depth_child_weight(self):

        gridsearch_params = [(max_depth, min_child_weight)
                              for max_depth in MAX_DEPTH_GRID for min_child_weight in MIN_CHILD_WEIGHT_GRID]
        update_params = self.params.copy()
        best_params = (update_params['max_depth'],
                       update_params['min_child_weight'])
        updated = False

        for max_depth, min_child_weight in gridsearch_params:
            print("CV with max_depth = %i, min_child_weight = %i" %
                  (max_depth, min_child_weight))
            # Update our parameters
            update_params['max_depth'] = max_depth
            update_params['min_child_weight'] = min_child_weight
            # run XGboost
            cv_results = xgb.cv(
                update_params,
                self.xtrain,
                num_boost_round=1000,
                seed=42,
                nfold=2,
                metrics={'mae'},
                early_stopping_rounds=10
            )
            # update best mae
            mean_mae = cv_results['test-mae-mean'].min()
            boost_rounds = cv_results['test-mae-mean'].argmin()
            print("\tMAE %.5f after %.5f rounds" % (mean_mae, boost_rounds))
            if mean_mae < self.best_mae:
                updated = True
                self.best_mae = mean_mae
                best_params = (max_depth, min_child_weight)

        if updated:
            self.params['max_depth'] = best_params[0]
            self.params['min_child_weight'] = best_params[1]
        print("Best params: max_depth: %i, min_child_weight: %i, MAE: %.5f" %
              (best_params[0], best_params[1], self.best_mae))
        print(f"\t New params are: {self.params}")

    def subsample_colsample(self):

        gridsearch_params = [(subsample, colsample) for subsample in [
                              i/10. for i in SUBSAMPLE_GRID] for colsample in [i/10. for i in COLSAMPLE_GRID]]
        update_params = self.params.copy()
        best_params = (update_params['subsample'],
                       update_params['colsample_bytree'])
        updated = False

        for subsample, colsample in gridsearch_params:
            print("CV with subsample = %f, colsample = %f" %
                  (subsample, colsample))
            # Update our parameters
            update_params['subsample'] = subsample
            update_params['colsample_bytree'] = colsample
            # run XGboost
            cv_results = xgb.cv(
                update_params,
                self.xtrain,
                num_boost_round=1000,
                seed=42,
                nfold=2,
                metrics={'mae'},
                early_stopping_rounds=10
            )
            # update best mae
            mean_mae = cv_results['test-mae-mean'].min()
            boost_rounds = cv_results['test-mae-mean'].argmin()
            print("\tMAE %.5f after %.5f rounds" % (mean_mae, boost_rounds))
            if mean_mae < self.best_mae:
                updated = True
                self.best_mae = mean_mae
                best_params = (subsample, colsample)

        if updated:
            self.params['subsample'] = best_params[0]
            self.params['colsample_bytree'] = best_params[1]
        print("Best params: subsample: %f, colsample: %f, MAE: %.5f" %
              (best_params[0], best_params[1], self.best_mae))
        print(f"\t New params are: {self.params}")

    def ETA(self):

        gridsearch_params = [np.round(i, 2) for i in ETA_GRID]

        update_params = self.params.copy()
        best_params = (update_params['eta'])
        updated = False

        for eta in gridsearch_params:
            print("CV with ETA = %.3f" % (eta))
            # Update our parameters
            update_params['eta'] = eta
            # run XGboost
            cv_results = xgb.cv(
                update_params, 
                self.xtrain,
                num_boost_round=3000,
                seed=42,
                nfold=2,
                metrics={'mae'},
                early_stopping_rounds=10
            )
            # update best mae
            mean_mae = cv_results['test-mae-mean'].min()
            boost_rounds = cv_results['test-mae-mean'].argmin()
            print("\tMAE %.5f after %.5f rounds" % (mean_mae, boost_rounds))
            if mean_mae < self.best_mae:
                updated = True
                self.best_mae = mean_mae
                best_params = (eta)
                
        if updated:    
            self.params['eta'] = best_params
        print("Best params: eta: %.3f, MAE: %.5f" % (best_params, self.best_mae))
        print(f"\t New params are: {self.params}")
    
    def save_params(self):

        with open(f'{config.OUTPUTS_PATH}{self.target}_XG_PARAMS.json', 'w') as f:
            json.dump(self.params, f, indent=4)


if __name__ == "__main__":


    hyper_train = HYPER_XG('dgp')
    hyper_train.initiate_model()
    hyper_train.depth_child_weight()
    hyper_train.subsample_colsample()
    hyper_train.ETA()
    hyper_train.save_params()

