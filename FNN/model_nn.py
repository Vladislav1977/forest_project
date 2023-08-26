from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import tqdm.notebook as tq
import torch






class NN_model:

    def __init__(self, init_layer, is_train=True, device="cpu"):
        self.device = device
        self.is_train = is_train
        self.model = nn.Sequential(nn.Linear(init_layer, 1200),
                                   nn.BatchNorm1d(1200),
                                   nn.LeakyReLU(),
                                   nn.Linear(1200, 200),
                                   nn.BatchNorm1d(200),
                                   nn.LeakyReLU(),
                                   nn.Linear(200, 7)
                                   ).to(device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=0.0005,
                                              weight_decay=1e-5)
            self.loss = nn.CrossEntropyLoss()

    def set_input(self, data):
        self.X = data[0].to(self.device)
        self.y_true = data[1].to(self.device)

    def predict(self, X_test):

        X_test = torch.tensor(X_test, dtype=torch.float32)
        self.model.eval()
        self.y_pred = self.model(X_test)

        self.pred_class = self.y_pred.argmax(dim=-1) + 1
        return self.pred_class

    def optimize_params(self):
        self.model.train()
        self.y_pred = self.model(self.X)

        self.pred_class = self.y_pred.argmax(dim=-1)

        self.loss_val = self.loss(self.y_pred, self.y_true)

        self.optimizer.zero_grad()
        self.loss_val.backward()
        self.optimizer.step()

    def evaluate(self):
        self.model.eval()

        with torch.no_grad():
            self.y_pred = self.model(self.X)
            self.loss_val = self.loss(self.y_pred, self.y_true)
            self.pred_class = self.y_pred.argmax(dim=-1)

    def save_model(self, scaler, filename, X_type):
        checkpoint = {
            "scaler": scaler,
            "model_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "X_type": X_type
        }
        torch.save(checkpoint, filename)
      #  print("Saving checkpoint")

    def load_model(self, checkpoint_file):

        print("Loading checkpoint")
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler = checkpoint["scaler"]
        self.X_type = checkpoint["X_type"]