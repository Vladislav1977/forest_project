import argparse
import os

import sys


from data.Dataset import MyDataset


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


class Config_parse:
    """This class defines options used during  training  time.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):

        # basic parameters
        parser.add_argument("X", type=str)
        parser.add_argument("scaler", type=str)
        parser.add_argument("--epoches", type=int, default=150)
        parser.add_argument("--save", action='store_true')
        parser.add_argument("--name", type=str)

        parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

        self.initialized = True
        return parser

    def gather_options(self):

        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            self.parser = parser

        # save and return the parser
        #self.parser = parser
        return self.parser.parse_args()

    def parse(self):
        opt = self.gather_options()

        self.opt = opt
        return self.opt



path_train = r"../dataset/train.csv"
path_test = r"../dataset/test.csv"

df = MyDataset(path_train, path_test)

CONFIG_NN = {"dataset_train":
              {"X_1": df.X_1,
               "X_2": df.X_2,
               "X_3": df.X_3,
               "y": df.y},
          "dataset_test":
              {"X_1": df.X_1_test,
               "X_2": df.X_2_test,
               "X_3": df.X_3_test,
               "Id": df.Id},

          "scaler":
              {"None": None,
               "std": StandardScaler(),
               "transform_1": ColumnTransformer(
                   [("std", StandardScaler(), slice(0, 10))], remainder='passthrough'),
               "transform_2": ColumnTransformer(
                   [("std", StandardScaler(), slice(0, 25))], remainder='passthrough'),
               "transform_3": ColumnTransformer(
                   [("std", StandardScaler(), slice(0, 33))], remainder='passthrough')},
          }

if __name__ == "__main__":
    print("os:", os.getcwd())
    print("path", sys.path)
    path_train = r"../dataset/train.csv"
    path_test = r"../dataset/test.csv"
    df = MyDataset(path_train, path_test)
    print(df.X_3.shape)