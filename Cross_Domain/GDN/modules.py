import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros

class GATLayer(MessagePassing):
    """
    Implementation of the Graph Attentional layer. Computes the message and propagate the message with Message Passing

    Parameters:
    -----------
        in_channels: int, input dimension of the layer
        out_channels: int, output dimension of the layer
        heads: int, number of heads in the layer
        concat: bool, if True, the output is the concatenation of the messages, else the output is the average of the messages
        negative_slope: float, negative slope of the leaky relu activation
        dropout: float, dropout probability
        bias: bool, if True, adds a bias to the output
        inter_dim: int, dimension of the intermediate layer
    """
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True, inter_dim=-1,**kwargs):
        super(GATLayer, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.node_dim = 0
        self.__alpha__ = None
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        self.att_i = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads*out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the parameters of the layer.
        """
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        zeros(self.att_em_i)
        zeros(self.att_em_j)
        zeros(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        """
        Forward pass of the layer. Computes the message and propagate the message with Message Passing
        """
        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x[1].size(self.node_dim))
        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index, return_attention_weights=return_attention_weights)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, size_i, embedding, edges, return_attention_weights):
        """
        Computes the message of the layer.
        """
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        if embedding is not None:
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]]
            embedding_i = embedding_i.unsqueeze(1).repeat(1,self.heads,1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1,self.heads,1)

            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)

        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)
        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)
        alpha = alpha.view(-1, self.heads, 1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes = size_i)

        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)


class GraphLayer(nn.Module):
    """
    Graph Attention Based Layer. Uses Batch Normalization and GAT to compute the embedding for the nodes. 

    Parameters:
    -----------
        in_channels: int, input dimension of the layer
        out_channels: int, output dimension of the layer
        heads: int, number of heads in the layer
    """
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GraphLayer, self).__init__()
        self.gnn = GATLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):
        """
        Forward pass of the layer.
        """
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
        out = self.bn(out)
        return self.relu(out)

class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        """
        Computes the forecasted values for the nodes.

        Parameters:
        -----------
        in_num: int, input dimension of the layer
        node_num: int, number of nodes in the graph
        layer_num: int, number of layers in the graph
        inter_num: int, dimension of the intermediate layer
        """
        super(OutLayer, self).__init__()
        layers = []
        for i in range(layer_num):
            # last layer, output shape:1
            if i==layer_num-1:
                if layer_num==1:
                    layers.append(nn.Linear(in_num , 1))
                else:
                    layers.append(nn.Linear(inter_num , 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                layers.append(nn.Linear( layer_in_num, inter_num ))
                layers.append(nn.BatchNorm1d(inter_num))
                layers.append(nn.ReLU())

        self.mlp = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass of the layer.
        """
        out = x
        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out
