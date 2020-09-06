import torch.nn as nn
import torch.nn.functional as F



def Net(in_num_of_features, out_num_of_features):
    num_features = 1024
    layers = []
    
    layers.append(nn.Linear(in_num_of_features, num_features))
    layers.append(nn.PReLU())
    layers.append(nn.BatchNorm1d(num_features=num_features))

    while (num_features >= 256):
        layers.append(nn.Linear(num_features, num_features // 2))
        layers.append(nn.PReLU())
        layers.append(nn.BatchNorm1d(num_features=num_features // 2))
        layers.append(nn.Dropout(0.2))

        num_features //= 2

    layers.append(nn.Linear(num_features, out_num_of_features))
    layers.append(nn.ReLU())

    return nn.Sequential(*layers)
