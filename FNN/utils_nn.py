
import torch
from torch.utils.data import Dataset


class NN_dataset(Dataset):

    def __init__(self, x_array, y_array=None):
        self.X = torch.tensor(x_array, dtype=torch.float32)
        self.y = torch.from_numpy(y_array)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] - 1

def accuracy_score(y_pred, y_true):

    return torch.sum(y_pred == y_true)
