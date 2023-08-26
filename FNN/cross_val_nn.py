import sys
sys.path.insert(1, "..")

import torch
from torch.utils.data import Dataset, DataLoader
from utils_nn import NN_dataset, accuracy_score

import tqdm as tq
from sklearn.model_selection import KFold


from config_nn import CONFIG_NN, Config_parse

import numpy as np
from model_nn import NN_model
from data.Dataset import MyDataset

import argparse






def cross_val_nn(opt):

    X, y = CONFIG_NN["dataset_train"][opt.X], CONFIG_NN["dataset_train"]["y"]
    scaler = CONFIG_NN["scaler"][opt.scaler]

    epoches, save, name = opt.epoches, opt.save, opt.name

    if scaler is not None:
        X_nn = scaler.fit_transform(X)
    else:
        X_nn = X

    init_layer = X_nn.shape[1]

    df = NN_dataset(X_nn, y)
    kf = KFold(n_splits=5, shuffle=True)

    kfold_acc = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        print(f"Fold {fold + 1}")
        print("-------")
        model_NN = NN_model(init_layer)
        # Define the data loaders for the current fold
        train_loader = DataLoader(
            dataset=df,
            batch_size=128,
            drop_last=False,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        )
        val_loader = DataLoader(
            dataset=df,
            batch_size=128,
            drop_last=False,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        )

        accuracy_stats = {'train': [], 'val': []}
        loss_stats = {'train': [], 'val': []}

        dls = {"train": (train_loader, len(train_idx)), "val": (val_loader, len(test_idx))}
        for epoch in tq.tqdm(range(epoches), position=0, leave=True):

            for phase, loader_len in dls.items():
                loader, size = loader_len
                run_loss = 0
                run_acc = 0

                for i, data in enumerate(loader):
                    model_NN.set_input(data)
                    if phase == "train":
                        model_NN.optimize_params()
                    elif phase == "val":
                        model_NN.evaluate()

                    run_loss += model_NN.loss_val
                    run_acc += accuracy_score(model_NN.pred_class, model_NN.y_true) / size

                loss_stats[phase].append((run_loss / (i + 1)).item())
                accuracy_stats[phase].append(run_acc.item())

            if epoch % 25 == 0:
                    print(
                        f'Epoch {epoch}: | Train Loss: {loss_stats["train"][epoch]} | Val Loss: {loss_stats["val"][epoch]}  | Train Acc: {accuracy_stats["train"][epoch]} | Val Acc:  {accuracy_stats["val"][epoch]}'
                    )

        kfold_acc.append(np.mean(accuracy_stats["val"][-10:]))
    if save:
        if name is None:
            raise ValueError("name must be specified.")
        model_NN.save_model(scaler, name, opt.X)
    print(kfold_acc)
    return kfold_acc

if __name__ == "__main__":

    opt = Config_parse().parse()
    cross_val_nn(opt)