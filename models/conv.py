import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.nn.norm import GraphNorm, PairNorm, MessageNorm, DiffGroupNorm, InstanceNorm, LayerNorm, GraphSizeNorm, MessageNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()
nn_act = torch.nn.ReLU() #ReLU()
F_act = F.relu

class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), nn_act, torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
        return out

    def message(self, x_j, edge_attr):
        if edge_attr is None:
            return F_act(x_j)
        return F_act(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)

        row, col = edge_index
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_attr, norm=norm) + F_act(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        if edge_attr is None:
            return norm.view(-1, 1) * F_act(x_j)
        return norm.view(-1, 1) * F_act(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_name = 'gin', norm_layer = 'batch_norm'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.norm_layer = norm_layer

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.graph_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_name == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_name == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_name))

            if norm_layer.split('_')[0] == 'batch':
                if norm_layer.split('_')[-1] == 'notrack':
                    self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim, track_running_stats=False, affine=False))
                else:
                    self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            elif norm_layer.split('_')[0] == 'instance':
                self.batch_norms.append(InstanceNorm(emb_dim))
            elif norm_layer.split('_')[0] == 'layer':
                self.batch_norms.append(LayerNorm(emb_dim))
            elif norm_layer.split('_')[0] == 'graph':
                self.batch_norms.append(GraphNorm(emb_dim))
            elif norm_layer.split('_')[0] == 'size':
                self.batch_norms.append(GraphSizeNorm())
            elif norm_layer.split('_')[0] == 'pair':
                self.batch_norms.append(PairNorm(emb_dim))
            elif norm_layer.split('_')[0] == 'group':
                self.batch_norms.append(DiffGroupNorm(emb_dim, groups=4))
            else:
                raise ValueError('Undefined normalization layer called {}'.format(norm_layer))
        if norm_layer.split('_')[1] == 'size':
            self.graph_size_norm = GraphSizeNorm()

    def forward(self, batched_data, encode_raw = True):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### computing input node embedding
        if encode_raw:
            h_list = [self.atom_encoder(x)]
            edge_attr = self.bond_encoder(edge_attr)
        else:
            h_list = [x]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            if self.norm_layer.split('_')[0] == 'batch':
                h = self.batch_norms[layer](h)
            else:
                h = self.batch_norms[layer](h, batch)
            if self.norm_layer.split('_')[1] == 'size':
                h = self.graph_size_norm(h, batch)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F_act(h)
                h = F.dropout(h, self.drop_ratio, training = self.training)
            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation, h_list


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_name = 'gin', norm_layer = 'batch_norm'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.norm_layer = norm_layer

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()
        self.graph_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_name == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_name == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_name))
            
            if norm_layer.split('_')[0] == 'batch':
                if norm_layer.split('_')[-1] == 'notrack':
                    self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim, track_running_stats=False, affine=False))
                else:
                    self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            elif norm_layer.split('_')[0] == 'instance':
                self.batch_norms.append(InstanceNorm(emb_dim))
            elif norm_layer.split('_')[0] == 'layer':
                self.batch_norms.append(LayerNorm(emb_dim))
            elif norm_layer.split('_')[0] == 'graph':
                self.batch_norms.append(GraphNorm(emb_dim))
            elif norm_layer.split('_')[0] == 'size':
                self.batch_norms.append(GraphSizeNorm())
            elif norm_layer.split('_')[0] == 'pair':
                self.batch_norms.append(PairNorm(emb_dim))
            elif norm_layer.split('_')[0] == 'group':
                self.batch_norms.append(DiffGroupNorm(emb_dim, groups=4))
            else:
                raise ValueError('Undefined normalization layer called {}'.format(norm_layer))
        if norm_layer.split('_')[1] == 'size':
            self.graph_size_norm = GraphSizeNorm()
        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), nn_act, \
                                                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), nn_act))


    def forward(self, batched_data, encode_raw = True):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        if encode_raw:
            h_list = [self.atom_encoder(x)]
            edge_attr = self.bond_encoder(edge_attr)
        else:
            h_list = [x]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]
            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            if self.norm_layer.split('_')[0] == 'batch':
                h = self.batch_norms[layer](h)
            else:
                h = self.batch_norms[layer](h, batch)
            if self.norm_layer.split('_')[1] == 'size':
                h = self.graph_size_norm(h, batch)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F_act(h)
                h = F.dropout(h, self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)


        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation, h_list



class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim, max_norm=1)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

            
    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim, max_norm=1)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding
