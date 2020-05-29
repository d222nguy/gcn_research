import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels =  in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()   
    def forward(self, x, adj):
        x_shortcut = x
        F1 = 20
        F2 = 20
        dropout = 0.5
        self.gc1 = GraphConvolution(self.in_channels, F1)
        self.gc2 = GraphConvolution(F1, F2)
        self.gc3 = GraphConvolution(F2, self.out_channels)

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, dropout, training = self.training)

        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, dropout, training = self.training)

        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, dropout, training = self.training)

        x = x + x_shortcut
        x = F.relu(x)
        return x