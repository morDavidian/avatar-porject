import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_num_of_features, out_num_of_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_num_of_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, out_num_of_features)
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
