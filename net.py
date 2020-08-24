import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_num_of_features, out_num_of_features):
        super(Net, self).__init__()
        num_features = 4096
        self.layers = []
        self.layers.append(dict(
            fc=nn.Linear(in_num_of_features, num_features//2),
            bn=nn.BatchNorm1d(num_features=num_features//2),
            dp=nn.Dropout(0.3)))
        num_features //= 2

        while (num_features >= 256):
            self.layers.append(dict(
                fc=nn.Linear(num_features, num_features//2),
                bn=nn.BatchNorm1d(num_features=num_features//2),
                dp=nn.Dropout(0.3)))
            num_features //= 2

        self.output_layer = nn.Linear(num_features, out_num_of_features) 
            
    def forward(self, x): 
        for l in self.layers:
            x = F.leaky_relu(l["bn"](l["fc"](x)))
        x = F.sigmoid(self.output_layer(x))
        return x
