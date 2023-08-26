import sys
sys.path.insert(1, "..")

import torch
from torch.utils.data import Dataset, DataLoader

import tqdm as tq
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from config_nn import CONFIG_NN, Config_parse

from model_nn import NN_model
from utils_nn import NN_dataset, accuracy_score



def train(opt):

    X = CONFIG_NN["dataset_train"][opt.X]
    y = CONFIG_NN["dataset_train"]["y"]
    scaler = CONFIG_NN["scaler"][opt.scaler]

    save = opt.save
    name = opt.name
    epoches = opt.epoches

    if scaler is not None:
        scaler.fit(X)
        X_nn = scaler.transform(X)
    else:
        X_nn = X

    init_layer = X_nn.shape[1]
    df = NN_dataset(X_nn, y)

    model_NN = NN_model(init_layer)
    # Define the data loaders for the current fold
    train_loader = DataLoader(
        dataset=df,
        batch_size=128,
        drop_last=False)

    accuracy_stats = []
    loss_stats = []

    size = len(X_nn)
    for epoch in tq.tqdm(range(epoches), position=0, leave=True):
        run_loss = 0
        run_acc = 0
        for i, data in enumerate(train_loader):
            model_NN.set_input(data)
            model_NN.optimize_params()

            run_loss += model_NN.loss_val
            run_acc += accuracy_score(model_NN.pred_class, model_NN.y_true) / size

        loss_stats.append((run_loss / (i + 1)).item())
        accuracy_stats.append(run_acc.item())

        if epoch % 25 == 0:
            print(
                f'Epoch {epoch}: | Train Loss: {loss_stats[epoch]}   | Train Acc: {accuracy_stats[epoch]}'
                )

    if opt.save:
        try:
            model_NN.save_model(scaler, name, opt.X)
            print("model saved")
        except AttributeError as AE:
            print("Save Error. Save Name should be defined (expected to be string)")
    return loss_stats, accuracy_stats



if __name__ == "__main__":

    opt = Config_parse().parse()
    #X = CONFIG_NN["dataset_train"][opt.X]
    #y = CONFIG_NN["dataset_test"]["y"]
    #scaler = CONFIG_NN["scaler"][opt.scaler]
    #train(X, y, scaler, opt.epoches, opt.save, opt.name)
    train(opt)
