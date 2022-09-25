import torch
import torch.nn.functional as F
from modules import GATConv

class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_features, num_classes):
        super().__init__()
        self.heads = 2
        self.conv1 = GATConv(num_features, hidden_features, heads = self.heads, concat = True, bias = True)
        self.conv2 = GATConv(self.heads*hidden_features, num_classes, heads = 1, concat = True, bias = True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)