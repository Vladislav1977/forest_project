import argparse
import sys

from data.Dataset import MyDataset
import argparse
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

import hyperopt as hp

from base_model import *

from catboost import CatBoostClassifier, Pool, metrics, cv
from sklearn.metrics import accuracy_score

from utils import *
path_train = r"./dataset/train.csv"
path_test = r"./dataset/test.csv"

df = MyDataset(path_train, path_test)

CONFIG = {"dataset":
              {"X_1": df.X_1,
               "X_2": df.X_2,
               "X_3": df.X_3,
               "y": df.y,
               "X_1_test": df.X_1_test,
               "X_2_test": df.X_2_test,
               "X_3_test": df.X_3_test,
               "Id": df.Id},

          "transform":
              {"None": None,
                "std": StandardScaler(),
               "transform_1" : ColumnTransformer(
                   [("std", StandardScaler(), slice(0, 10))], remainder='passthrough'),
               "transform_2" : ColumnTransformer(
                   [("std", StandardScaler(), slice(0, 25))], remainder='passthrough'),
               "transform_3" : ColumnTransformer(
                   [("std", StandardScaler(), slice(0, 33))], remainder='passthrough')},
          "model":
              {"logreg": LogisticRegression(penalty="l1",
                                            solver="saga",
                                            C=0.8666666666666667,
                                            max_iter=5000),
               "SVM": SVC(C=10, gamma=0.2782559402207126),
               "gradboost": CatBoostClassifier(depth = 7,
                                               l2_leaf_reg=0.5552399305644247,
                                               learning_rate=0.1798420272630059,
                                               iterations=985,
                                               eval_metric="Accuracy",
                                               random_seed=42,
                                               verbose=False,
                                               loss_function="MultiClass"),
               "extree": ExtraTreesClassifier(
                   n_estimators=189,
                   min_samples_split=2,
                   min_samples_leaf=1)},
          "params":
              {"logreg": {"model__C": hyperopt.hp.lognormal('something', 0, 1),
                          "model__penalty": hyperopt.hp.choice("penalty", ["l1", "l2", "none"])},
               "SVM": {'model__C': hyperopt.hp.loguniform("C", np.log(pow(10, -3)), np.log(pow(10, 3))),
                       "model__gamma": hyperopt.hp.choice(
                           "gamma", [hyperopt.hp.loguniform("gamma_log", np.log(pow(10, -3)), np.log(pow(10, 3))), "scale", "auto"])},
               "gradboost": {'model__l2_leaf_reg': hyperopt.hp.loguniform('l2_leaf_reg', -1, 2),
                             'model__learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
                             'model__depth': hyperopt.hp.randint('depth', 4, 10)},
               "extree": {"model__n_estimators": hyperopt.hp.randint('depth', 50, 550),
                          "model__min_samples_split":hyperopt.hp.randint('min_split', 2, 10),
                          "model__min_samples_leaf": hyperopt.hp.randint('min_label', 1, 10)}
               }

          }



class Ml_app(object):

    def __init__(self):
        parser = argparse.ArgumentParser(

            usage='''ML_app <command> [<args>]

The most commonly used git commands are:
   cross_val     cross validation
   hypersearch   Bayes Hypersearch tuning
   predict       Predict on test set
''')
        parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def parser_init(self, predict=False):
        parser = argparse.ArgumentParser(
            description='Record changes to the repository')
        # prefixing the argument with -- means it's optional
        parser.add_argument('dataset', choices=["X_1", "X_2", "X_3"])
        parser.add_argument('scaler', choices=["None", "std", "transform_1", "transform_2", "transform_3"])
        parser.add_argument('model', choices=["logreg", "SVM", "gradboost", "extree"])
        parser.add_argument('--max_eval', type=int, default=50) #for hypersearch
        if predict:
            parser.add_argument("name", type=str, help='Define file name')


        # now that we're inside a subcommand, ignore the first
        # TWO argvs, ie the command (git) and the subcommand (commit)
        args, unknown = parser.parse_known_args(sys.argv[2:])
        self.model_name = args.model

        self.model = CONFIG["model"][args.model]
        self.scaler = CONFIG["transform"][args.scaler]
        self.X = CONFIG["dataset"][args.dataset]
        if predict:
            self.X_test = CONFIG["dataset"][args.dataset + "_test"]
            self.Id = CONFIG["dataset"]["Id"]
            self.name = args.name
        self.y = CONFIG["dataset"]["y"]
        self.max_eval = args.max_eval
        self.args = args
        if len(unknown) > 0:
            model_arg = parse_unknown(unknown)
            self.model.set_params(**model_arg)
#            print(self.model)

        self.base = Base_Model(self.model,
                          self.X,
                          self.y,
                          self.scaler)

    def cross_val(self):

        mlflow.start_run()

        self.parser_init()
        mlflow.log_param("dataset", self.args.dataset)
        mlflow.log_param("scaler", self.args.scaler)

        print("Running cross validation")

        return self.base.cross_val()

    def hypersearch(self):

        self.parser_init()
        print("Running hypersearch")
        params = CONFIG["params"][self.model_name]

        self.base.hypersearch(params, self.max_eval)

    def predict(self):

        self.parser_init(predict=True)
        print("Making prediction")
        self.base.predict(self.X_test, self.Id, self.name)








if __name__ == '__main__':
    Ml_app()