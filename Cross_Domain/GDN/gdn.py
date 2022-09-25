import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.component_logger import component_logger as logger
from modules import GraphLayer, OutLayer

class GDN(nn.Module):
    """
    GDN: Graph Deviation Network; Implements a Graph-based Anomaly Detection Anomaly Detection Model

    Graph Deviation Network fixes the attributes as Nodes, and corresponding the windowed sequences as features for these nodes. 
    These features are then encoded into a hidden representation using Graph Attention based intermediate layers. The hidden representation is then used to forecast
    the next time-step for the node/feature. The MSE between the forecast and ground truth is used to train the model.  
        
    Attributes
    ----------
        edge_index : 2-D array; edge_index.shape = [2, num_edges]
            It is the adjaceny matrix of the graph.

        node_num : integer
            The number of nodes in the graph. It is equal to number of features/ dimensions in dataset.

        dim : integer
            The dimension of the node embedding.

        input_dim : integer
            The dimension of the node input. It is equal to the window/ sequence length.

        out_layer_num : integer
            The number of penultimate output layers in the graph.

        out_layer_inter_dim : integer
            The dimension of the penultimate output layers in the graph.
        
        topk : integer
            The number of top-k nodes to be used in the graph. 
            Used to learn the structure of the graph, to mask certain nodes and their corresponding features while calculating node embeddings.
    """
    def __init__(self, edge_index, node_num, dim, input_dim, out_layer_num, out_layer_inter_dim, topk):
        super(GDN, self).__init__()
        self.edge_index = edge_index
        embed_dim = dim
        self.topk = topk
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)
        self.gnn_layer = GraphLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1)
        self.out_layer = OutLayer(dim, node_num, out_layer_num, inter_num = out_layer_inter_dim)
        self.dp = nn.Dropout(0.2)
        self.node_embedding = None
        self.learned_graph = None
        self.batch_edge_index = None
        self.init_params()
    
    def init_params(self):
        """Initialize the embedding of the model with kaiming uniform initialization.
        """
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))
        
    
    def get_batch_edge_index(self, org_edge_index, batch_num, node_num):
        """
            gets the edge index for a batch of nodes.
            Parameters
            ----------
                org_edge_index: edge index for fully connected graph (except self-connections).
                batch_num: batch number
                node_num: number of nodes in the batch
            Return
            -------
                edge_index: edge index for batch of nodes.
        """
        try:
            edge_index = org_edge_index.clone().detach()
            edge_num = org_edge_index.shape[1]
            batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

            for i in range(batch_num):
                batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num
        except Exception as error:
            logger.log("Failed to get batch edge index: {}".format(error))
            exit()

        return batch_edge_index.long()

    def forward(self, data):
        """
            Forward pass of the model. Given the input data, it returns the output of the GDN model using graph attention based forecasting method.
        """

        device = data.device
        x = data.clone().detach()
        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()
        gcn_outs = []

        # Get the embedding of the nodes
        all_embeddings = self.embedding(torch.arange(node_num).to(device))
        weights_arr = all_embeddings.detach().clone()
        all_embeddings = all_embeddings.repeat(batch_num, 1)
        
        weights = weights_arr.view(node_num, -1)

        # Graph Structure Learning. Use cosine similarity between the embeddings to find the top-k nodes.
        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
        cos_ji_mat = cos_ji_mat / normed_mat
        
        topk_num = self.topk
        topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

        self.learned_graph = topk_indices_ji
        
        # Compute the edge values between the nodes using attention mechanism/ graph attention based layers (gnn_layer).
        gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
        gated_j = topk_indices_ji.flatten().unsqueeze(0)
        gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
        batch_gated_edge_index = self.get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
        self.batch_edge_index = batch_gated_edge_index
        gcn_out = self.gnn_layer(x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)
        gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)
        
        indexes = torch.arange(0,node_num).to(device)

        # Output is the multiplication of the node embedding and the features.
        out = torch.mul(x, self.embedding(indexes))

        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)

        # Forecasting the next time-step.
        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)

        return out


