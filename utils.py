

from catboost import Pool

import numpy as np

from hyperopt import space_eval, STATUS_OK
import ast


def hyperopt_objective_extr(params, model, cv, X, y):

    model = model.set_params(**params)
    model_name = model["model"].__class__.__name__

    if (model_name.find("CatBoostClassifier") == -1):

        cv_data = cv(model, X, y, scoring="accuracy")
        best_acc = cv_data.mean()
        return {"loss": 1 - best_acc, "params": params, "status": STATUS_OK}
    else:
        if model["scaler"] is not None:
            X = model["scaler"].fit_transform(X)

        cv_data = cv(Pool(X, y),
                     model["model"].get_params(),
                     nfold=5,
                     logging_level='Silent')
        best_acc = np.max(cv_data['test-Accuracy-mean'])
        best_iteration = cv_data["test-Accuracy-mean"].argmax()

        return {"loss": 1 - best_acc, "params": params, "status": STATUS_OK, "iter": best_iteration}


def string_parse(str_arg):
    try:
        str_arg = ast.literal_eval(str_arg)
        return str_arg
    except:
        return str_arg

def parse_unknown(args):
    key = list(map(lambda x: x[2:], args[::2]))
    val = list(map(lambda x: string_parse(x), args[1::2]))
    return dict(zip(key, val))






