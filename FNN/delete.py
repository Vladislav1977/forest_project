import os

import torch
import torch.nn as nn

from model_nn import *

#model = NN_model(75)
#
# = "./weights/test"

#checkpoint = torch.load(path, map_location="cpu")
#print(checkpoint["model_dict"]["0.weight"])model = NN_model(75)
#
# = "./weights/test"

#checkpoint = torch.load(path, map_location="cpu")
#print(checkpoint["model_dict"]["0.weight"])

import sys
for i in sys.path:
    print(i)

print("getcwd", os.getcwd())

model = NN_model(75)

path = "./weights/test"

checkpoint = torch.load(path, map_location="cpu")
#print(checkpoint["model_dict"]["0.weight"])
#model.load_model(path)

#print(model.X_type)

#from data.Dataset import MyDataset

