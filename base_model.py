from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

from catboost import Pool, cv


from functools import partial

import numpy as np
import pandas as pd

from utils import *
import hyperopt
from hyperopt import space_eval, STATUS_OK






class Base_model:

    def __init__(self, model, X, y, scaler):

        self.model = Pipeline(
                    [("scaler", scaler), ("model", model)])
  #      print(self.model)
        model_name = self.model["model"].__class__.__name__
        self.X = X
        self.y = y
        self.cv = cross_val_score if (model_name.find("CatBoostClassifier") == -1) else cv


    def fit(self):

        self.model.fit(self.X, self.y)

    def predict(self, X_test, Id, name):

        self.model.fit(self.X, self.y)

        y_pred = self.model.predict(X_test).flatten()

        df_subm = pd.DataFrame({"Cover_Type": y_pred, "Id": Id}).set_index("Id")
        df_subm.to_csv(name)

    def cross_val(self):

        if self.cv.__name__ == 'cross_val_score':
            cv = self.cv(self.model, self.X, self.y, scoring="accuracy")
            print("Mean CV:", cv.mean())
            return None
        else:
            if self.model["scaler"] is not None:
                self.X = self.model["scaler"].fit_transform(self.X)

            cv = self.cv(Pool(self.X, self.y),
                         self.model["model"].get_params(),
                         nfold=5,
                         logging_level='Silent')
            max_val = cv["test-Accuracy-mean"].max()
            max_est = cv["test-Accuracy-mean"].argmax() + 1
            message = f"Max. mean accuracy: {max_val}, best_est: {max_est}"
            print(message)
            return None

    def hypersearch(self, params, max_evals=50):

        trials_ex = hyperopt.Trials()

        best = hyperopt.fmin(
            partial(hyperopt_objective_extr, model=self.model,
                    cv=self.cv,
                    X=self.X,
                    y=self.y),
            space=params,
            algo=hyperopt.tpe.suggest,
            max_evals=max_evals,
            trials=trials_ex
        )
        #print("Trial:", trials_ex.results)
        #print("Best:", best)
        print("Best params:", space_eval(params, best))