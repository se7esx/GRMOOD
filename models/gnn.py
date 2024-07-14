import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from .conv import GNN_node, GNN_node_Virtualnode

from utils.FGW import *
from utils.mol_utils import *
device = torch.device('cuda', 1)
class GNN(torch.nn.Module):
    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, gnn_type = 'gin', drop_ratio = 0.5, graph_pooling = "max", norm_layer = 'batch_norm',template_number=1):
        '''
            num_tasks (int): number of labels to be predicted
        '''

        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.template_number = template_number
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        ### GNN to generate node embeddings
        gnn_name = gnn_type.split('-')[0]
        if 'virtual' in gnn_type:
            self.graph_encoder = GNN_node_Virtualnode(num_layer, emb_dim, JK = 'last', drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name, norm_layer = norm_layer)
        else:
            self.graph_encoder = GNN_node(num_layer, emb_dim, JK = 'last', drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name, norm_layer = norm_layer)
        ### Poolinwg function to generate whole-graph embeddings
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        rep_dim = emb_dim
        self.predictor = torch.nn.Sequential(torch.nn.Linear(rep_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Dropout(), torch.nn.Linear(2*emb_dim, self.num_tasks))

        self.predictor_with_distance =  torch.nn.Sequential(torch.nn.Linear(self.template_number, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Dropout(), torch.nn.Linear(2*emb_dim, self.num_tasks))
    def forward(self, batched_data,template_dataloader, encode_raw = True):
        if template_dataloader is None:
            h_node, _ = self.graph_encoder(batched_data, encode_raw)
            h_graph = self.pool(h_node, batched_data.batch)
            return self.predictor(h_graph), h_graph
        else:
            
            h_node, _ = self.graph_encoder(batched_data, encode_raw)
            template_Data = next(iter(template_dataloader)).to(device)
            h_template,_ = self.graph_encoder(template_Data, encode_raw)
            #print(h_node)
            batch_dense_x, batch_dense_x_disable, batch_dense_enable_adj, batch_dense_disable_adj, batch_node_mask = \
                convert_to_batch(batched_data.batch, h_node, batched_data.edge_index, batched_data.edge_attr,
                                        augment_mask=None)
            template_dense_x, template_dense_x_disable, template_dense_enable_adj, template_dense_disable_adj, template_node_mask = \
                convert_to_batch(template_Data.batch, h_template, template_Data.edge_index, template_Data.edge_attr,
                                        augment_mask=None)
            list_of_tensors = th.split(batch_dense_enable_adj, 1, dim=0)
            list_of_tensors = [t.squeeze(dim=0).to('cpu') for t in list_of_tensors]
            list_of_features = th.split(batch_dense_x, 1, dim=0)
            list_of_features = [t.squeeze(dim=0).to('cpu') for t in list_of_features]
            list_h = [th.ones(C.shape[0], dtype=th.float64, device='cpu') / C.shape[0] for C in
                      list_of_tensors]
            template_of_tensors = th.split(template_dense_enable_adj, 1, dim=0)
            template_of_tensors = [t.squeeze(dim=0).to('cpu') for t in template_of_tensors]
            template_of_features = th.split(template_dense_x, 1, dim=0)
            template_of_features = [t.squeeze(dim=0).to('cpu') for t in template_of_features]
            template_h = [th.ones(C.shape[0], dtype=th.float64, device='cpu') / C.shape[0] for C in
                      template_of_tensors]
            #print(list_of_tensors[0].shape)
            #print(list_of_features[0].shape)
            #print(template_of_tensors[0].shape)
            #print(template_of_features[0].shape)
            
            h_graph = parallelized_get_features(self.template_number, list_of_tensors, list_of_features, list_h, template_of_tensors, template_of_features,template_h,
                                      th.tensor(0.5)).to(device)
            
            h_graph = h_graph.to(torch.float32)
            return self.predictor_with_distance(h_graph), h_graph
