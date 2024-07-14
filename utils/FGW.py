import numpy as np
import torch as th
import sys
sys.path.append("/home/wangyili/shenxu/ood-and-diffusion-1.0/utils")

import utils.GW_utils as GW_utils
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from torch_geometric.nn import GINConv
#alpha->fixed , structure-> adj, device->cpu,dtype = th.float64
learn_alpha = False
alpha = th.tensor([0.5], requires_grad=False, dtype=th.float64, device='cpu')
FGW_solver = (lambda C1,C2,F1,F2,M,h1,h2,alpha,learn_alpha : GW_utils.parallel_fused_gromov_wasserstein2_learnablealpha(
            C1=C1, C2=C2, F1=F1, F2=F2, M=M, p=h1, q=h2, alpha=alpha, compute_gradients = True, learn_alpha=learn_alpha))
def get_features_by_input_cpu( C, F, h, list_Cbar, list_Fbar, list_hbar, alpha, local_device='cpu'):
    res = []
    localK = len(list_Cbar)
    dim_features = F.shape[-1]

    F2 = F ** 2
    ones_source = th.ones((F.shape[0], dim_features), dtype=th.float64, device=local_device)
    
    I = th.eye(C.shape[0], dtype=th.float64, device=local_device)
    localC = C - I
    
    for i in range(localK):
        shape_atom = list_Cbar[i].shape[0]
        ones_target = th.ones((dim_features, shape_atom), dtype=th.float64, device=local_device)
        first_term = F2.to(th.float64) @ ones_target
        second_term = ones_source @ (list_Fbar[i].to(th.float64) ** 2).T
        
        Mi = first_term + second_term - 2 * F @ list_Fbar[i].T

        # res.append(GW_utils.parallel_fused_gromov_wasserstein2_learnablealpha(C1=localC, C2=list_Cbar[i], F1=F, F2=list_Fbar[i], M=Mi, p=h, q=list_hbar[i], alpha=alpha, learn_alpha=False ))
        res.append(FGW_solver(C1=localC, C2=list_Cbar[i], F1=F, F2=list_Fbar[i], M=Mi, h1=h, h2=list_hbar[i],
                                   alpha=alpha, learn_alpha=False))
     
    return res
def get_features_by_input_gpu(C, F, h, list_Cbar, list_Fbar, list_hbar, list_Mi, alpha):
    res = []

    Katoms = len(list_Cbar)

    for i in range(Katoms):
        M = list_Mi[i]
        res.append(
            GW_utils.parallel_fused_gromov_wasserstein2_learnablealpha(C1=C, C2=list_Cbar[i], F1=F, F2=list_Fbar[i],
                                                                       M=M, p=h, q=list_hbar[i], alpha=alpha,
                                                                       learn_alpha=False))
    return res

def compute_pairwise_euclidean_distance(list_F, list_Fbar, detach=True):
    list_Mik = []
    dim_features = list_F[0].shape[-1]
    for F in list_F:
        list_Mi = []
        F2 = F ** 2
        ones_source = th.ones((F.shape[0], dim_features), dtype=th.float64, device='cpu')
        for Fbar in list_Fbar:
            shape_atom = Fbar.shape[0]
            ones_target = th.ones((dim_features, shape_atom), dtype=th.float64, device='cpu')
            first_term = F2 @ ones_target
            second_term = ones_source @ (Fbar ** 2).T
            Mi = first_term + second_term - 2 * F @ Fbar.T
            if detach:
                list_Mi.append(Mi.detach().cpu())
            else:
                list_Mi.append(Mi)

        list_Mik.append(list_Mi)
    return list_Mik
def parallelized_get_features(Katoms,list_C, list_F, list_h, list_Cbar, list_Fbar, list_hbar, alpha, evaluate=False,
                              n_jobs=2):
    """
    list_G: list of input structures
    list_h: list of corresponding masses
    """
    ValFunction = GW_utils.ValFunction
    # check_gpu_memory_usage()

    features = th.zeros((len(list_C), Katoms), dtype=th.float64, device='cpu')

    with th.no_grad():

        res_by_input = Parallel(n_jobs=n_jobs)(
            delayed(get_features_by_input_cpu)(list_C[i], list_F[i], list_h[i], list_Cbar, list_Fbar,
                                                    list_hbar, alpha) for i in range(len(list_C)))

    if not evaluate:

        for idx_res in range(len(res_by_input)):
            for idx_atom in range(Katoms):
                fgw, gh, ghbar, gC, gCbar, gF, gFbar, _ = res_by_input[idx_res][idx_atom]
                fgw = GW_utils.set_gradients(ValFunction, fgw, (
                list_h[idx_res], list_hbar[idx_atom], list_C[idx_res], list_Cbar[idx_atom], list_F[idx_res],
                list_Fbar[idx_atom]),
                                             (gh, ghbar.to('cpu'), gC, gCbar.to('cpu'), gF,
                                              gFbar.to('cpu')))
                features[idx_res, idx_atom] = fgw
    else:
        for idx_res in range(len(res_by_input)):
            for idx_atom in range(Katoms):
                fgw = res_by_input[idx_res][idx_atom]
                features[idx_res, idx_atom] = fgw

    return features

