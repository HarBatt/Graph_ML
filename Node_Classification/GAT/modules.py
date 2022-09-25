import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros

seed = 26
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads, concat, bias, edge_dim= None, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim = 0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.edge_dim = edge_dim

        # Used to split the feature vector into heads.
        self.lin_src = Linear(in_channels, out_channels*heads, bias=False)
        self.lin_dst = self.lin_src

        # The learnable parameters to compute attention coefficients.
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, out_channels*heads, bias=False)
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))

        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)
        
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads*out_channels))
        
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)


    def forward(self, x, edge_index, edge_attr = None, return_attention_weights = None):
        heads, out_channels = self.heads, self.out_channels

        # Divide x into chunks of size heads after passing through MLP.
        x_src = self.lin_src(x).view(-1, heads, out_channels)
        x_dst = self.lin_dst(x).view(-1, heads, out_channels)

        x = (x_src, x_dst)
        
        # We only want to add self-loops for nodes that appear both as source and target nodes:
        num_nodes = x[0].size(0)
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes = num_nodes)

        # Next, we compute node-level attention coefficients, both for source and target nodes (if present):
        # Shape of x_src: (batch_size, heads, out_channels)
        # Shape of att_src: (1, heads, out_channels)
        # Shape of att_src: (batch_size, heads)
        # Shape of att_dst: (batch_size, heads)

        alpha_src  = (x_src*self.att_src).sum(dim= -1)
        alpha_dst  = (x_dst*self.att_dst).sum(dim= -1)
    
        alpha = (alpha_src, alpha_dst)

        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        out = self.propagate(edge_index, x=x, alpha=alpha)

        if self.concat:
            out = out.view(-1, heads*out_channels)
        
        else:
            out = out.view(-1, out_channels)
    
        if self.bias is not None:
            out = out + self.bias
        
        if return_attention_weights:
            return out, (edge_index, alpha)

        return out


    def edge_update(self, alpha_j, alpha_i, edge_attr, index, ptr):
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation

        alpha = alpha_j + alpha_i
        # Working on edges
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr*self.att_edge).sum(dim= -1)
            alpha = alpha + alpha_edge
        
        # Finally, we apply softmax to each attention coefficient to get normalized attention weights:
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, index, ptr)
        alpha = F.dropout(alpha, p= 0.2)
        return alpha

    
    def message(self, x_j, alpha):
        current_alpha = alpha.unsqueeze(-1)*x_j
        return current_alpha
    
    def __rep__(self):
        return "({}, in_channels: {}, out_channels: {}, heads: {})".format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)