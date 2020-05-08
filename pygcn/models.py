import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        #self.gc2 = GraphConvolution(nhid, nclass) We don't want a network that shallow
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        n_layer = 15
        
        first_hid = F.dropout(F.relu(self.gc1(x, adj)), self.dropout, training = self.training)
        out = F.dropout(F.relu(self.gc2(first_hid, adj)), self.dropout, training = self.training)
        for _ in range(n_layer):
            out = F.dropout(F.relu(self.gc2(out, adj)), self.dropout, training = self.training)
        out = out + first_hid
        out = self.gc3(out, adj)
        return F.log_softmax(out, dim=1)
