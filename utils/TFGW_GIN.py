import numpy as np
import torch as th
import os
from tqdm import tqdm
import pickle
import utils_networks
import GW_utils 
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from torch_geometric.nn import GINConv

#%%

str_to_metrics = {'accuracy' : (lambda y_true,y_pred : accuracy_score(y_true,y_pred))}

def get_features_by_input_gpu(C, F, h, list_Cbar, list_Fbar, list_hbar, list_Mi, alpha, alpha_mode):
    res = []
    
    Katoms= len(list_Cbar)
    if alpha_mode == 'fixed':
        for i in range(Katoms):
            M = list_Mi[i]
            res.append(GW_utils.parallel_fused_gromov_wasserstein2_learnablealpha(C1=C, C2=list_Cbar[i], F1=F, F2=list_Fbar[i], M=M, p=h, q=list_hbar[i], alpha=alpha, learn_alpha=False ))
    elif alpha_mode == 'learnable_shared':
        for i in range(Katoms):
            M = list_Mi[i]
            res.append(GW_utils.parallel_fused_gromov_wasserstein2_learnablealpha(C1=C, C2=list_Cbar[i], F1=F, F2=list_Fbar[i], M=M, p=h, q=list_hbar[i], alpha=alpha, learn_alpha=True ))
    return res

    

class GIN_FGWmachine():
    
    def __init__(self,
                 graph_mode:str,
                 input_shape:int,
                 Katoms:int, 
                 n_labels:int, 
                 alpha:float, 
                 learn_hbar:bool,
                 experiment_repo:str, 
                 gin_net_dict:dict,
                 gin_layer_dict:dict,
                 clf_net_dict:dict,
                 skip_first_features:bool=False,
                 dtype = th.float64,
                 device='cpu'):
        """
        Implementation of our TFGW model with GIN pre-processing.
        
        Parameters
        ----------
        graph_mode : str
            in ['ADJ', 'SP']. Coincides with the data provided in the run_TFGW_GIN.py file.
            If set to 'SP' we used tuple to represent graphs as the GNN needs ADJ will our TFGW layer need SP.
        input_shape : int
            dimension of the features in the input.
        Katoms : int
            number of templates in the Wasserstein layer
        n_labels : int
            number of classes in the dataset.
        alpha : float
            trade-off parameter for FGW.
            alpha == -1: learn it
            otherwise the method support a fix alpha between 0 and 1.
        learn_hbar: bool
            either to learn the weights of the templates or not.
        experiment_repo : str
            repository to save the experiment during training.
        gin_net_dict : dict
            Dictionary containing parameters for the global architecture of GIN.
            Must contain the keys:
                'hidden_dim' : dimension of the hidden layer validated in {16,32,64} depending on datasets.
                                The output dimension of the layer is the same than the hidden dimension.
                'num_hidden' : the number of hidden layers in each MLP.                                 
        gin_layer_dict : dict
            Dictionary containing parameters for the GIN layers. The parameters (eps, train_eps) are fixed to (0, False) as suggested by authors.
            Must contain the key
                'num_layers' : number of GIN layers in the architecture (strictly positive integer)
        clf_net_dict : dict
            Dictionary containing parameters for the MLP leading to label prediction
            Must contain the keys
                'hidden_dim' :(int) dimension of the hidden layer (fixed to 128)
                'num_hidden' :(int) number of hidden layers in the architecture
                'dropout' :(float) dropout rate to use
        skip_first_features : bool, optional
            Either to skip the input features or not in the concatenation of all GIN layer outputs. (see Jumping Knowledge Networks)
            The default is False.
        dtype : TYPE, optional
            DESCRIPTION. The default is th.float64.
        device : TYPE, optional
            DESCRIPTION. The default is 'cpu'.
        """
        assert np.all( [s in gin_net_dict.keys() for s in ['hidden_dim', 'num_hidden'] ])
        assert np.all( [s in gin_layer_dict.keys() for s in ['num_layers']])
        assert np.all( [s in clf_net_dict.keys() for s in ['hidden_dim', 'num_hidden', 'dropout']] )
        assert gin_layer_dict['num_layers'] > 0
        self.graph_mode = graph_mode
        self.input_shape = input_shape
        self.Katoms = Katoms
        self.n_labels = n_labels
        self.device = device
        self.gin_net_dict = gin_net_dict
        self.gin_layer_dict = gin_layer_dict   
        self.skip_first_features = skip_first_features # if set to True when aggregate_gin_layers is True: skip input features in the aggregated template features.
        
        self.classification_metrics = {'accuracy' : (lambda y_true,y_pred : accuracy_score(y_true,y_pred))}
        
        self.experiment_repo = experiment_repo
        if not os.path.exists(self.experiment_repo):
            os.makedirs(self.experiment_repo)
        self.Cbar, self.Fbar, self.hbar = None, None, None
        self.dtype = dtype
        # Instantiate network for GIN embeddings
        self.GIN_layers = th.nn.ModuleList()
        
        for layer in range(gin_layer_dict['num_layers']):
            if layer == 0:
                local_input_shape = self.input_shape
            else:
                local_input_shape = gin_net_dict['hidden_dim']
            MLP = utils_networks.MLP_batchnorm(
                local_input_shape, 
                gin_net_dict['hidden_dim'], 
                gin_net_dict['hidden_dim'], 
                gin_net_dict['num_hidden'], 'relu',  device=self.device, dtype=self.dtype)
            GIN_layer = GINConv(MLP, eps= 0., train_eps=False).to(self.device)
            
            self.GIN_layers.append(GIN_layer)
                 

        # Instantiate network for classification
        if clf_net_dict['dropout'] != 0.:
            self.clf_Net = utils_networks.MLP_dropout(
                input_dim = self.Katoms, 
                output_dim = self.n_labels,
                hidden_dim = clf_net_dict['hidden_dim'],
                num_hidden = clf_net_dict['num_hidden'],
                output_activation= 'linear', 
                dropout = clf_net_dict['dropout'],
                skip_first = True,
                device=self.device, dtype=self.dtype)
        else:
            self.clf_Net = utils_networks.MLP(
                self.Katoms, 
                self.n_labels,
                clf_net_dict['hidden_dim'],
                clf_net_dict['num_hidden'],
                'linear', device=self.device, dtype=self.dtype)
      
        self.learn_hbar = learn_hbar
        
        if alpha == -1:
            self.alpha = th.tensor([0.5], requires_grad=True, dtype=dtype, device=self.device)
            self.alpha_mode = 'learnable_shared'
        else:
            self.alpha = th.tensor([alpha], requires_grad=False, dtype=dtype, device=self.device)
            self.alpha_mode = 'fixed'
        self.loss = th.nn.CrossEntropyLoss().to(self.device)
    
    def init_parameters_with_aggregation(self, 
                                         list_Satoms:list, 
                                         init_mode_atoms:str,
                                         graphs:list=None, features:list=None, 
                                         labels:list=None, atoms_projection:str='clipped'):
        assert len(list_Satoms)==self.Katoms
        
       
        # Handle FGW templates
        self.Cbar = []
        self.Fbar = []
        self.hbar = []
        for S in list_Satoms:
            x = th.ones(S, dtype=self.dtype, device=self.device) /S
            x.requires_grad_(self.learn_hbar)
            self.hbar.append(x)
    
        if 'sampling_supervised' in init_mode_atoms:
            print('init_mode_atoms = sampling_supervised' )
            # If not enough samples have the required shape within a label
            # we get the samples within the label and create perturbated versions of observed graphs
            # Then we do a forward pass within the GIN networks to get embedded features
            # of sampled graphs.
            shapes = th.tensor([C.shape[0] for C in graphs])
            unique_atom_shapes, counts = np.unique(list_Satoms,return_counts=True)
            unique_labels = th.unique(labels)
            idx_by_labels = [th.where(labels==label)[0] for label in unique_labels]
            
            for i,shape in enumerate(unique_atom_shapes):
                r = counts[i]
                perm = th.randperm(unique_labels.shape[0])
                count_by_label = r// unique_labels.shape[0]
                for i in perm:
                    i_ = i.item()
                    shapes_label = shapes[idx_by_labels[i_]]
                    shape_idx_by_label = th.where(shapes_label==shape)[0]
                    print('shape_idx_by_label (S=%s) / %s'%(shape, shape_idx_by_label))
                    if shape_idx_by_label.shape[0] == 0:
                        print('no samples for shape S=%s -- computing kmeans on features within the label'%shape)
                        stacked_features = th.cat(features)
                        print('stacked features shape:', stacked_features.shape)
                        km = KMeans(n_clusters = shape, init='k-means++', n_init=10, random_state = 0)
                        km.fit(stacked_features)
                        
                        F_clusters = th.tensor(km.cluster_centers_, dtype=self.dtype, device=self.device)
                        print('features from clusters:', F_clusters.shape)
                    try:
                        print('found samples of shape =%s / within label  =%s'%(shape, i_))
                        sample_idx = np.random.choice(shape_idx_by_label.numpy(), size=min(r,count_by_label), replace=False)
                        print('sample_idx:', sample_idx)
                        for idx in sample_idx:
                            localC = graphs[idx_by_labels[i_][idx]].clone().to(self.device)
                            localC.requires_grad_(True)
                            localF = features[idx_by_labels[i_][idx]].clone().to(self.device)
                            #localF.requires_grad_(True)
                            self.Cbar.append(localC)      
                            self.Fbar.append(localF)
                    except:
                        print(' not enough found samples of shape =%s / within label  =%s'%(shape, i_))
                        try:
                            sample_idx = np.random.choice(shape_idx_by_label.numpy(), size=min(r,count_by_label), replace=True)
                            print('sample idx:', sample_idx)
                            for idx in sample_idx:
                                C = graphs[idx_by_labels[i_][idx]].clone().to(self.device)
                                noise_distrib_C = th.distributions.normal.Normal(loc=0., scale= C.std() + 1e-05)
                                noise_C = noise_distrib_C.rsample(C.size()).to(self.device)
                                if th.all(C == C.T):
                                    noise_C = (noise_C + noise_C)/2
                                F = features[idx_by_labels[i_][idx]].clone().to(self.device)
                                noise_distrib_F = th.distributions.normal.Normal(loc=th.zeros(F.shape[-1]), scale= F.std(axis=0) + 1e-05)
                                noise_F = noise_distrib_F.rsample(th.Size([shape])).to(self.device)
                                new_C = C + noise_C
                                new_C.requires_grad_(True)                                             
                                new_F = F + noise_F
                                #new_F.requires_grad_(True)
                                self.Cbar.append(new_C)
                                self.Fbar.append(new_F)
                        except:
                            print(' NO sample found with proper shape within label = %s'%i_)
                            # We generate a random graph from the distribution of graphs within this label
                            # Randomly over C
                            # by kmeans over F
                            local_idx = idx_by_labels[i_]
                            means_C = th.tensor([graphs[idx].mean() for idx in local_idx])
                            distrib_C = th.distributions.normal.Normal(loc= means_C.mean(), scale= means_C.std()+ 1e-05)
                            for _ in range(min(r,count_by_label)):
                                C = distrib_C.rsample((shape, shape)).to(self.device)
                                C = (C + C.T) / 2.
                                if atoms_projection == 'clipped':
                                    new_C = ( C - C.min())/ (C.max() - C.min())
                                elif atoms_projection == 'nonnegative':
                                    new_C = C.clamp(0)
                                new_C.requires_grad_(True)                                             
                                self.Cbar.append(new_C)
                                new_F = F_clusters.clone().to(self.device)
                                self.Fbar.append(new_F)
                            
                        continue
                            
                # Embed template features through GIN layers
                # stack features for the sampled initial templates at each GIN layers :
                with th.no_grad():
                    for idx_atom in range(self.Katoms):
                        F = self.Fbar[idx_atom].to(self.device)
                        processedC = self.Cbar[idx_atom].to(self.device)
                        edge_index = th.argwhere( processedC == 1.).T
                        list_embedded_features  = []
                        if not self.skip_first_features:
                            list_embedded_features.append(F)
                        embedded_features_layered = self.GIN_layers[0](x=F, edge_index=edge_index)
                        list_embedded_features.append(embedded_features_layered)
                        for local_layer in range(1, self.gin_layer_dict['num_layers']):
                            embedded_features_layered = self.GIN_layers[local_layer](x=embedded_features_layered, edge_index=edge_index)
                            list_embedded_features.append(embedded_features_layered)
                        embedded_features = th.cat(list_embedded_features, dim=1)
                        embedded_features.requires_grad_(True)
                        self.Fbar[idx_atom] = embedded_features
                    print('processed Fbar shapes:', [F.shape for F in self.Fbar])
        self.atoms_params = [*self.Cbar, *self.Fbar]
        if self.learn_hbar:
            self.atoms_params += [*self.hbar]
        if self.alpha_mode in ['learnable_shared', 'learnable_indep']:
            self.atoms_params += [self.alpha]
        self.params =  self.atoms_params + list(self.GIN_layers.parameters()) + list(self.clf_Net.parameters())
        
        #self.layered_dim_features = [self.Fbar[self.Katoms * local_layer].shape[-1] for local_layer in range(self.gin_layer_dict['num_layers'] + 1)]
        self.shape_atoms = [C.shape[0] for C in self.Cbar]
        print('---  model initialized  ---')
        print('Fbar dims:', [F.shape for F in self.Fbar])
        #print('sanity check params (shape, requires grad, device):', [(param.shape, param.requires_grad, param.device) for param in self.params])
        print('GIN_layers:', self.GIN_layers.parameters)
        print('Clf:', self.clf_Net.parameters)
    
    def compute_pairwise_euclidean_distance(self, list_F, list_Fbar, detach = True):
        list_Mik = []
        dim_features = list_F[0].shape[-1]
        for F in list_F:
            list_Mi = []
            F2 = F**2
            ones_source = th.ones((F.shape[0], dim_features), dtype=self.dtype, device=self.device)
            for Fbar in list_Fbar:
                
                shape_atom = Fbar.shape[0]
                ones_target = th.ones((dim_features, shape_atom), dtype=self.dtype, device=self.device)
                first_term = F2 @ ones_target
                second_term = ones_source @ (Fbar**2).T
                Mi = first_term + second_term - 2* F @ Fbar.T
                if detach:
                    list_Mi.append(Mi.detach().cpu())
                else:
                    list_Mi.append(Mi)
                    
            list_Mik.append(list_Mi)
        return list_Mik
    
    def get_features_by_input_cpu(self, C, F, h, list_Cbar, list_Fbar, list_hbar, alpha, local_device='cpu'):
        res = []
        localK = len(list_Cbar)
        dim_features = F.shape[-1]
        F2 = F**2
        ones_source = th.ones((F.shape[0], dim_features), dtype=self.dtype, device=local_device)
        if self.graph_mode == 'ADJ':
            I = th.eye(C.shape[0], dtype=self.dtype, device=local_device)
            localC = C - I
        else:
            localC = C
        if self.alpha_mode == 'fixed':
            for i in range(localK):
                shape_atom = list_Cbar[i].shape[0]
                ones_target = th.ones((dim_features, shape_atom), dtype=self.dtype, device=local_device)
                first_term = F2 @ ones_target
                second_term = ones_source @ (list_Fbar[i]**2).T
                Mi = first_term + second_term - 2* F @ list_Fbar[i].T
            
                #res.append(GW_utils.parallel_fused_gromov_wasserstein2_learnablealpha(C1=localC, C2=list_Cbar[i], F1=F, F2=list_Fbar[i], M=Mi, p=h, q=list_hbar[i], alpha=alpha, learn_alpha=False ))
                res.append(self.FGW_solver(C1=localC, C2=list_Cbar[i], F1=F, F2=list_Fbar[i], M=Mi, h1=h, h2=list_hbar[i], alpha=alpha, learn_alpha=False ))

        elif self.alpha_mode == 'learnable_shared':
            for i in range(localK):
                shape_atom = list_Cbar[i].shape[0]
                ones_target = th.ones((dim_features, shape_atom), dtype=self.dtype, device=local_device)
                first_term = F2 @ ones_target
                second_term = ones_source @ (list_Fbar[i]**2).T
                Mi = first_term + second_term - 2* F @ list_Fbar[i].T
            
                #res.append(GW_utils.parallel_fused_gromov_wasserstein2_learnablealpha(C1=localC, C2=list_Cbar[i], F1=F, F2=list_Fbar[i], M=Mi, p=h, q=list_hbar[i], alpha=alpha, learn_alpha=True ))
                res.append(self.FGW_solver(C1=localC, C2=list_Cbar[i], F1=F, F2=list_Fbar[i], M=Mi, h1=h, h2=list_hbar[i], alpha=alpha, learn_alpha=True ))

        return res
    
    
    
    def parallelized_get_features(self, list_C, list_F, list_h, list_Cbar, list_Fbar, list_hbar, alpha, evaluate=False, n_jobs=1):
        """
        list_G: list of input structures
        list_h: list of corresponding masses
        """
        
        #check_gpu_memory_usage()

        features = th.zeros((len(list_C), self.Katoms), dtype=self.dtype, device=self.device)
        
                    
        with th.no_grad():
            
            if self.device == 'cpu':
                res_by_input = Parallel(n_jobs=n_jobs)(delayed(self.get_features_by_input_cpu)(list_C[i], list_F[i], list_h[i], list_Cbar, list_Fbar, list_hbar, alpha) for i in range(len(list_C)))
            else:
                list_Mik = self.compute_pairwise_euclidean_distance(list_F, list_Fbar)
                #print('MIK:', [[M.device for M in list_Mi] for list_Mi in list_Mik])
                if self.graph_mode == 'ADJ':
                    local_list_C = []
                    for C in list_C:
                        I = th.eye(C.shape[0], dtype=self.dtype, device=self.device)                        
                        localC = C - I
                        local_list_C.append(localC.detach().cpu())
                else:
                    local_list_C = [C.detach().cpu() for C in list_C]
                local_list_F = [F.detach().cpu() for F in list_F]
                local_list_h = [h.detach().cpu() for h in list_h]
                local_list_Cbar = [C.detach().cpu() for C in list_Cbar]
                local_list_Fbar = [F.detach().cpu() for F in list_Fbar]
                local_list_hbar = [h.detach().cpu() for h in list_hbar]
                local_alpha = alpha.detach().cpu()
                res_by_input = Parallel(n_jobs=n_jobs)(delayed(get_features_by_input_gpu)(
                    local_list_C[i], local_list_F[i], local_list_h[i], local_list_Cbar, local_list_Fbar, local_list_hbar, list_Mik[i], local_alpha, self.alpha_mode) for i in range(len(local_list_C)))
            
        if not evaluate:
        
            if self.alpha_mode =='learnable_shared':
                
                for idx_res in range(len(res_by_input)):    
                    for idx_atom in range(self.Katoms):
                        
                        fgw, gh, ghbar, gC, gCbar, gF, gFbar, galpha = res_by_input[idx_res][idx_atom]
                        fgw = GW_utils.set_gradients(self.ValFunction, fgw.to(self.device), 
                                                     (list_h[idx_res], list_hbar[idx_atom], list_C[idx_res], list_Cbar[idx_atom], list_F[idx_res], list_Fbar[idx_atom], alpha), 
                                                     (gh.to(self.device), ghbar.to(self.device).to(self.device), gC.to(self.device), gCbar.to(self.device), gF.to(self.device), gFbar.to(self.device), galpha.to(self.device)))
                        features[idx_res, idx_atom] = fgw
            elif self.alpha_mode == 'fixed':
                
                for idx_res in range(len(res_by_input)):    
                    for idx_atom in range(self.Katoms):
                        
                        fgw, gh, ghbar, gC, gCbar, gF, gFbar, _ = res_by_input[idx_res][idx_atom]
                        fgw = GW_utils.set_gradients(self.ValFunction, fgw, (list_h[idx_res], list_hbar[idx_atom], list_C[idx_res], list_Cbar[idx_atom], list_F[idx_res], list_Fbar[idx_atom]),
                                                     (gh, ghbar.to(self.device), gC, gCbar.to(self.device), gF, gFbar.to(self.device)))
                        features[idx_res, idx_atom] = fgw
        else:    
            for idx_res in range(len(res_by_input)):    
                for idx_atom in range(self.Katoms):                    
                    fgw = res_by_input[idx_res][idx_atom]
                    features[idx_res, idx_atom] = fgw
        
        
        return features
    
    
    def set_model_to_train(self):
        self.GIN_layers.train()
        self.clf_Net.train()
        # with gradients computation within the solver
        self.FGW_solver = (lambda C1,C2,F1,F2,M,h1,h2,alpha,learn_alpha : GW_utils.parallel_fused_gromov_wasserstein2_learnablealpha(
            C1=C1, C2=C2, F1=F1, F2=F2, M=M, p=h1, q=h2, alpha=alpha, compute_gradients = True, learn_alpha=learn_alpha))
        
    def set_model_to_eval(self):
        self.GIN_layers.eval()
        self.clf_Net.eval()
        # without gradients computation within the solver
        self.FGW_solver = (lambda C1,C2,F1,F2,M,h1,h2,alpha,learn_alpha : GW_utils.parallel_fused_gromov_wasserstein2_learnablealpha(
            C1=C1, C2=C2, F1=F1, F2=F2, M=M, p=h1, q=h2, compute_gradients = False, alpha=alpha))
    
    def GIN_forward(self, batch_graphs, batch_features, batch_shapes, cumsum_shapes):
        
        if self.graph_mode == 'ADJ':                        
            processed_batch_graphs = th.block_diag(*[C for C in batch_graphs])                 
        elif self.graph_mode == 'SP':                        
            processed_batch_graphs = th.block_diag(*[C[1] for C in batch_graphs])                 
        batch_edge_index = th.argwhere(processed_batch_graphs == 1.).T
        processed_batch_features = th.cat([F for F in batch_features])
        
        layered_batch_features = []
        if not self.skip_first_features:
            layered_batch_features.append(processed_batch_features)
        
        batch_embedded_features = self.GIN_layers[0](x=processed_batch_features, edge_index=batch_edge_index)
        
        layered_batch_features.append(batch_embedded_features)
            
        for layer in range(1, self.gin_layer_dict['num_layers']):
            batch_embedded_features = self.GIN_layers[layer](x=batch_embedded_features, edge_index=batch_edge_index)
            layered_batch_features.append(batch_embedded_features)
        batch_embedded_features = th.cat(layered_batch_features, dim = 1)                                        
        batch_embedded_features_uncat =  [batch_embedded_features[cumsum_shapes[k] : cumsum_shapes[k + 1], :] for k in range(len(batch_shapes))]
        return batch_embedded_features_uncat 
    
    def fit(self, 
            model_name:str, 
            X_train:list, F_train:list, y_train:list, 
            X_val:list, F_val:list, y_val:list, 
            X_test:list, F_test:list, y_test:list,
            atoms_projection:str, lr:float, batch_size:int, supervised_sampler:bool, 
            epochs:int, val_timestamp:int, algo_seed:int,
            track_templates:bool=False, verbose:bool=False, n_jobs:int=None):
        th.manual_seed(algo_seed)
        np.random.seed(algo_seed)
        if not n_jobs is None:
            self.ValFunction = GW_utils.ValFunction

        if self.graph_mode == 'ADJ':
            h_train = [th.ones(C.shape[0], dtype=self.dtype, device=self.device)/C.shape[0] for C in X_train]
        elif self.graph_mode == 'SP':
            h_train = [th.ones(C[0].shape[0], dtype=self.dtype, device=self.device)/C[0].shape[0] for C in X_train]
        
        n_train = y_train.shape[0]
        for param in [self.Cbar, self.Fbar, self.alpha]:
            assert (param != None)
        
        if not (X_val is None):
            if self.graph_mode == 'ADJ':
                h_val = [th.ones(C.shape[0], dtype=self.dtype, device=self.device)/C.shape[0] for C in X_val]
                h_test = [th.ones(C.shape[0], dtype=self.dtype, device=self.device)/C.shape[0] for C in X_test]
            elif self.graph_mode == 'SP':
                h_val = [th.ones(C[0].shape[0], dtype=self.dtype, device=self.device)/C[0].shape[0] for C in X_val]
                h_test = [th.ones(C[0].shape[0], dtype=self.dtype, device=self.device)/C[0].shape[0] for C in X_test]
            
            sets = ['train', 'val', 'test']
            best_val_acc = - np.inf
            best_val_acc_train_acc = - np.inf
            
        else:
            raise 'provide a validation'
        self.log = {'train_cumulated_batch_loss':[]}
        for metric in list(self.classification_metrics.keys()) + ['epoch_loss']:
            for s in sets:
                self.log['%s_%s'%(s, metric)]=[]
        
        if track_templates:
            self.log_templates = {}
            for label in range(self.n_labels):
                for s in sets:
                    self.log_templates['%s_mean_dists_label%s'%(s, label)] = []
                    self.log_templates['%s_std_dists_label%s'%(s, label)] = []
                    
        
        self.optimizer = th.optim.Adam(params=self.params, lr=lr, betas=[0.9, 0.99])
        print('alpha requires grad : ', self.alpha.requires_grad)    
        
        if n_train <= batch_size:
            batch_by_epoch = 1
            batch_size = n_train
            print('batch size bigger than #samples > batch_size set to ', batch_size)
        else:
            batch_by_epoch = n_train//batch_size +1
        y_train_ = y_train.detach().cpu()
        if self.graph_mode == 'ADJ':
            shapes_train_ = th.tensor([C.shape[0] for C in X_train], device='cpu', dtype=th.int32)
        elif self.graph_mode == 'SP':
            shapes_train_ = th.tensor([C[0].shape[0] for C in X_train], device='cpu', dtype=th.int32)
        if supervised_sampler:
            unique_labels = th.unique(y_train_)
            n_labels = unique_labels.shape[0]
            train_idx_by_labels = [th.where(y_train_==label)[0] for label in unique_labels]
            labels_by_batch = batch_size // n_labels
        
        self.set_model_to_train()
        for e in tqdm(range(epochs), desc='epochs'):
            
            cumulated_batch_loss=0.
            #for batch_i in tqdm(range(batch_by_epoch),desc='epoch %s'%e):
            for batch_i in range(batch_by_epoch):
                
                self.optimizer.zero_grad()
                if not supervised_sampler:
                    batch_idx = np.random.choice(range(n_train), size=batch_size, replace=False)
                else:
                    r = batch_size
                    for idx_label,label  in enumerate(np.random.permutation(unique_labels)):
                        
                        local_batch_idx = np.random.choice(train_idx_by_labels[label], size=min(r,labels_by_batch), replace=False)
                        if idx_label ==0:
                            batch_idx = local_batch_idx
                        else:
                            batch_idx = np.concatenate([batch_idx,local_batch_idx])
                        r -= labels_by_batch

                batch_graphs = [X_train[idx] for idx in batch_idx]
        
                batch_shapes = shapes_train_[batch_idx]
                cumsum_shapes = th.cat([th.tensor([0]),batch_shapes]).cumsum(dim=0)
                #print('cumsum_shapes:', cumsum_shapes)
                batch_features = [F_train[idx] for idx in batch_idx]                
                batch_masses = [h_train[idx] for idx in batch_idx]
                batch_embedded_features_uncat = self.GIN_forward(batch_graphs, batch_features, batch_shapes, cumsum_shapes)
                # GNN filters
                
                if self.graph_mode == 'ADJ':
                    dist_features = self.parallelized_get_features(
                        batch_graphs, batch_embedded_features_uncat, batch_masses, 
                        self.Cbar, self.Fbar, self.hbar, self.alpha, n_jobs=n_jobs)
                elif self.graph_mode == 'SP':
                    dist_features = self.parallelized_get_features(
                        [C[0] for C in batch_graphs], batch_embedded_features_uncat, batch_masses, 
                        self.Cbar, self.Fbar, self.hbar, self.alpha, n_jobs=n_jobs)
                
                batch_pred = self.clf_Net(dist_features)
                batch_loss = self.loss(batch_pred, y_train[batch_idx])
                cumulated_batch_loss += batch_loss.item()
                batch_loss.backward()                
                self.optimizer.step()
                
                with th.no_grad():    
                    
                    for C in self.Cbar:
                        if atoms_projection =='clipped':
                            C[:] = C.clamp(min=0,max=1)
                        else:
                            C[:] = C.clamp(min=0)

                    if self.alpha.requires_grad:
                        self.alpha[:] = self.alpha.clamp(min=0., max=1. )

                    if self.learn_hbar:
                        for h in self.hbar:
                            h[:] = GW_utils.probability_simplex_projection(h)
            self.log['train_cumulated_batch_loss'].append(cumulated_batch_loss)
                #print('current lr with scheduler:', self.scheduler.get_last_lr())
            if (((e %val_timestamp) ==0) and e>0) or (e == (epochs - 1)):
                with th.no_grad():
                    self.set_model_to_eval()
                    
                    if self.learn_hbar:
                        pruned_Cbar, pruned_Fbar, pruned_hbar = self.prune_templates()
                    else:
                        # unchanged
                        pruned_Cbar, pruned_Fbar, pruned_hbar = self.Cbar, self.Fbar, self.hbar
                    
                    if self.device== 'cpu':
                        features_train, pred_train, y_pred_train, loss_train, res_train = self.evaluate_fullbatch(X_train, F_train, h_train, y_train, pruned_Cbar, pruned_Fbar, pruned_hbar, n_jobs)
                        features_val, pred_val, y_pred_val, loss_val, res_val = self.evaluate_fullbatch(X_val, F_val, h_val, y_val, pruned_Cbar, pruned_Fbar, pruned_hbar, n_jobs)
                        features_test, pred_test, y_pred_test, loss_test, res_test = self.evaluate_fullbatch(X_test, F_test, h_test, y_test, pruned_Cbar, pruned_Fbar, pruned_hbar, n_jobs)
                    else:
                        features_train, pred_train, y_pred_train, loss_train, res_train = self.evaluate_minibatch(X_train, F_train, h_train, y_train,  pruned_Cbar, pruned_Fbar, pruned_hbar, batch_size, n_jobs)
                        features_val, pred_val, y_pred_val, loss_val, res_val = self.evaluate_minibatch(X_val, F_val, h_val, y_val, pruned_Cbar, pruned_Fbar, pruned_hbar, batch_size, n_jobs)
                        features_test, pred_test, y_pred_test, loss_test, res_test = self.evaluate_minibatch(X_test, F_test, h_test, y_test, pruned_Cbar, pruned_Fbar, pruned_hbar, batch_size, n_jobs)
                    
                    if verbose:
                        if self.learn_hbar:
                            print('hbar:', self.hbar)
                        if 'learnable' in self.alpha_mode:
                            print('alpha: %s / requires_grad: %s'%(self.alpha, self.alpha.requires_grad))
                        print('epoch= %s / loss_train = %s / res_train = %s / loss_val =%s/ res_val =%s'%(e,loss_train.item(),res_train,loss_val.item(),res_val))
                    self.log['train_epoch_loss'].append(loss_train.item())
                    self.log['val_epoch_loss'].append(loss_val.item())
                    self.log['test_epoch_loss'].append(loss_test.item())
                    for metric in self.classification_metrics.keys():
                        self.log['train_%s'%metric].append(res_train[metric])
                        self.log['val_%s'%metric].append(res_val[metric])
                        self.log['test_%s'%metric].append(res_test[metric])
                    
                    str_log = self.experiment_repo+'/%s_training_log.pkl'%model_name
                    pickle.dump(self.log, open(str_log,'wb'))
                    if best_val_acc <= res_val['accuracy']: #Save model with best val acc assuring increase of train acc
                        
                        best_val_acc = res_val['accuracy']
                        if best_val_acc_train_acc <= res_train['accuracy']:
                            best_val_acc_train_acc = res_train['accuracy']
                            str_file = self.experiment_repo+'/%s_best_val_accuracy_increasing_train_accuracy.pkl'%model_name
                            full_dict_state = {'epoch' : e,
                                               'GIN_params':self.GIN_layers.state_dict(),
                                               'clf_params':self.clf_Net.state_dict(),
                                               'atoms_params':[param.clone().detach().cpu() for param in self.atoms_params]
                                               }
                            pickle.dump(full_dict_state, open(str_file, 'wb'))
                                                   
                    if track_templates:
                        for label in th.unique(y_train):
                            label_ = int(label.item())
                            idx_train = th.where(y_train==label)[0]
                            subfeatures_train = features_train[idx_train]
                            idx_val = th.where(y_val==label)[0]
                            subfeatures_val = features_val[idx_val]
                            self.log_templates['train_mean_dists_label%s'%label_].append(subfeatures_train.mean(axis=0))
                            self.log_templates['train_std_dists_label%s'%label_].append(subfeatures_train.std(axis=0))
                            self.log_templates['val_mean_dists_label%s'%label_].append(subfeatures_val.mean(axis=0))
                            self.log_templates['val_std_dists_label%s'%label_].append(subfeatures_val.std(axis=0))

                            if verbose:
                                print('[TRAIN] label = %s / mean dists: %s / std dists: %s'%(label,self.log_templates['train_mean_dists_label%s'%label_][-1], self.log_templates['train_std_dists_label%s'%label_][-1]))
                            #print('[VAL] label = %s / mean dists: %s / std dists: %s'%(label,self.log_weights['val_mean_dists_label%s'%label_][-1], self.log_weights['val_std_dists_label%s'%label_][-1]))
                        str_log = self.experiment_repo+'/%s_tracking_templates_log.pkl'%model_name
                        pickle.dump(self.log_templates, open(str_log,'wb'))
                    
                # after evaluation, make the model trainable again
                self.set_model_to_train()
                
    def evaluate_fullbatch(self, list_C:list, list_F:list, list_h:list, list_y:list,
                           list_Cbar:list, list_Fbar:list, list_hbar:list,
                           n_jobs:int=None, return_node_embeddings:bool=False):
        #print('--- evaluate current model ---')
        self.set_model_to_eval()
        
        with th.no_grad():
            if self.graph_mode == 'ADJ':
                batch_shapes = [C.shape[0] for C in list_C]
            elif self.graph_mode == 'SP':
                batch_shapes = [C[0].shape[0] for C in list_C]
            cumsum_shapes = th.cat([th.tensor([0]),batch_shapes]).cumsum(dim=0)
            
            batch_embedded_features_uncat =  self.GIN_forward(list_C, list_F, batch_shapes, cumsum_shapes)
        
            if self.graph_mode == 'ADJ':
                dist_features = self.parallelized_get_features(
                    list_C, batch_embedded_features_uncat, list_h, 
                    list_Cbar, list_Fbar, list_hbar, self.alpha, evaluate=True, n_jobs=n_jobs)
            elif self.graph_mode == 'SP':
                dist_features = self.parallelized_get_features(
                    [C[0] for C in list_C], batch_embedded_features_uncat, list_h, 
                    list_Cbar, list_Fbar, list_hbar, self.alpha, evaluate=True, n_jobs=n_jobs)
            pred = self.clf_Net(dist_features)
            loss = self.loss(pred, list_y)
            y_pred = pred.argmax(1)
            y_ = list_y.detach().numpy()
            y_pred_ = y_pred.detach().numpy()
            res = {}
            for metric in self.classification_metrics.keys():
                res[metric] = self.classification_metrics[metric](y_,y_pred_)
        if not return_node_embeddings:
            return dist_features, pred, y_pred, loss, res
        
        else:
            return dist_features, batch_embedded_features_uncat, pred, y_pred, loss, res
        
    def evaluate_minibatch(self, list_C:list, list_F:list, list_h:list, list_y:list,
                           list_Cbar:list, list_Fbar:list, list_hbar:list,
                           n_jobs:int=2, batch_size:int = 128):
        #print('--- evaluate current model ---')
        self.set_model_to_eval()

        with th.no_grad():
            
            full_dist_features = []
            len_ = len(list_C)
            n_splits = len_ // batch_size + 1
            full_idx = np.arange(len_)
            # get distance features by split batches
            for k in range(n_splits):
                idx_batch = full_idx[k * batch_size : (k + 1) * batch_size]
                #print('idx_batch:', idx_batch[0], idx_batch[-1])
                local_list_C = [list_C[idx] for idx in idx_batch]
                local_list_F = [list_F[idx] for idx in idx_batch]
                local_list_h = [list_h[idx] for idx in idx_batch]
                
                if self.graph_mode == 'ADJ':
                    batch_shapes = [C.shape[0] for C in local_list_C]
                elif self.graph_mode =='SP':
                    batch_shapes = [C[0].shape[0] for C in local_list_C]
                cumsum_shapes = th.tensor([0] + batch_shapes).cumsum(dim=0)
                
                batch_embedded_features_uncat = self.GIN_forward(local_list_C, local_list_F, batch_shapes, cumsum_shapes)
                if self.graph_mode == 'ADJ':
                    dist_features = self.parallelized_get_features(
                        local_list_C, batch_embedded_features_uncat, local_list_h, 
                        list_Cbar, list_Fbar, list_hbar, self.alpha, evaluate=True, n_jobs=n_jobs)
                elif self.graph_mode == 'SP':
                    dist_features = self.parallelized_get_features(
                        [C[0] for C in local_list_C], batch_embedded_features_uncat, local_list_h, 
                        list_Cbar, list_Fbar, list_hbar, self.alpha, evaluate=True, n_jobs=n_jobs)
    
                full_dist_features.append(dist_features)
            dist_features = th.cat(full_dist_features, dim=0)
            pred = self.clf_Net(dist_features)
            loss = self.loss(pred, list_y)
            y_pred = pred.argmax(1)
            y_ = list_y.detach().cpu().numpy()
            y_pred_ = y_pred.detach().cpu().numpy()
            res = {}
            for metric in self.classification_metrics.keys():
                res[metric] = self.classification_metrics[metric](y_,y_pred_)                
        return dist_features, pred, y_pred, loss, res
    
    def prune_atoms(self):
        pruned_hbar = []
        pruned_Cbar = []
        pruned_Fbar = []
        for i, h in enumerate(self.hbar):
            nonzero_idx = th.argwhere(h>0)[:, 0]
            pruned_hbar.append(h[nonzero_idx])
            pruned_Cbar.append(self.Cbar[i][nonzero_idx, :][:, nonzero_idx])
            pruned_Fbar.append(self.Fbar[i][nonzero_idx,:])
        return pruned_Cbar, pruned_Fbar, pruned_hbar
        
    def evaluate_templates(self):
        #print('--- evaluate current model ---')
        self.set_model_to_eval()
        
        with th.no_grad():
            # compute distances between templates:
            dist_features = th.zeros((self.Katoms, self.Katoms), dtype=self.dtype, device=self.device)
            for i in range(self.Katoms-1):
                for j in range(i+1, self.Katoms):
                    res = self.get_features_by_input_cpu(
                        self.Cbar[i], self.Fbar[i], self.hbar[i], 
                        [self.Cbar[j]], [self.Fbar[j]], [self.hbar[j]], self.alpha)
                    dist_features[i, j] = res[0]
                    dist_features[j, i] = res[0]
                                    
            
            pred = self.clf_Net(dist_features)
            y_pred = pred.argmax(1)
        return dist_features, pred, y_pred
        
    def load(self, model_name:str, dtype:type=th.float64):
        str_file = '%s/%s_best_val_accuracy_increasing_train_accuracy.pkl'%(self.experiment_repo, model_name)
        full_dict_state = pickle.load(open(str_file, 'rb'))
        self.Cbar = []
        self.Fbar = []
        self.hbar = []
        self.ValFunction = GW_utils.ValFunction
        
        for p in full_dict_state['atoms_params'][:self.Katoms]:  # atoms structure
            C = p.clone().to(self.device)
            C.requires_grad_(False)
            self.Cbar.append(C)
        for p in full_dict_state['atoms_params'][self.Katoms : 2 * self.Katoms]:  # atoms structure
            F = p.clone().to(self.device)
            F.requires_grad_(False)
            self.Fbar.append(F)
        self.dim_features = self.Fbar[-1].shape[-1]
        self.shape_atoms = [C.shape[0] for C in self.Cbar]
        if self.learn_hbar:
            for p in full_dict_state['atoms_params'][2 * self.Katoms : 3 * self.Katoms]:  # atoms structure
                h = p.clone().to(self.device)
                h.requires_grad_(False)
                self.hbar.append(h)
        else:
            for s in self.shape_atoms:
                h = th.ones(s, dtype=self.dtype)/s
                self.hbar.append(h)
        if self.alpha_mode == 'learnable_shared':
            self.alpha = full_dict_state['atoms_params'][-1].clone().to(self.device)
        self.clf_Net.load_state_dict(full_dict_state['clf_params'])
        self.GIN_layers.load_state_dict(full_dict_state['GIN_params'])
        
        print('[SUCCESSFULLY LOADED] ',str_file)
    