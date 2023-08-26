import sys
sys.path.insert(1, "..")

import pandas as pd
from config_nn import CONFIG_NN, Config_parse
import argparse
from model_nn import NN_model
import torch
from model_nn import NN_model
from utils_nn import NN_dataset


def test_NN(opt):

    checkpoint = torch.load(opt.model_path, map_location="cpu")

    X = CONFIG_NN["dataset_test"][checkpoint["X_type"]]
    Id = CONFIG_NN["dataset_test"]["Id"]

    model = NN_model(X.shape[1])
    model.load_model(opt.model_path)

    if model.scaler is not None:
        X = model.scaler.transform(X)

    y_pred_NN = model.predict(X)

    df_lr_subm = pd.DataFrame({"Cover_Type": y_pred_NN.numpy()}, index=Id)
    df_lr_subm.to_csv(opt.name)

if __name__=="__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_path", type=str)
    parser.add_argument("name", type=str)
    opt = parser.parse_args()

    test_NN(opt)