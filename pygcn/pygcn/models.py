  
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.gcx = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        nlayers = 8
        x = F.dropout(F.relu(self.gc1(x, adj)), self.dropout, training = self.training)
        for i in range(nlayers):
            x = F.dropout(F.relu(self.gcx(x, adj)), self.dropout, training = self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
