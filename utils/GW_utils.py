"""
Solvers from the Python Optimal Transport library 
https://pythonot.github.io/ (version POT=0.8.1)
 
adapted for the integration of (Fused) Gromov-Wasserstein in TFGW layers
"""

import numpy as np
import ot
import torch as th
from ot.bregman import sinkhorn
from ot.utils import dist, UndefinedParameter, list_to_array
from ot.optim import cg
from ot.lp import emd_1d, emd
from ot.utils import check_random_state
from ot.backend import get_backend
from ot.gromov import init_matrix, gwloss, gwggrad

#%%

    


def parallel_gromov_wasserstein2(C1, C2, p, q, loss_fun='square_loss', log=False, armijo=False, G0=None, **kwargs):
    r"""
    Returns the gromov-wasserstein discrepancy between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`
    The function solves the following optimization problem:
    .. math::
        GW = \min_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}
    Where :
    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity
      matrices
    Note that when using backends, this loss function is differentiable wrt the
    marices and weights for quadratic loss using the gradients from [38]_.
    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,)
        Distribution in the source space.
    q :  array-like, shape (nt,)
        Distribution in the target space.
    loss_fun :  str
        loss function used for the solver either 'square_loss' or 'kl_loss'
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research. Else closed form is used.
        If there are convergence issues use False.
    Returns
    -------
    gw_dist : float
        Gromov-Wasserstein distance
    log : dict
        convergence information and Coupling marix
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    .. [13] Mémoli, Facundo. Gromov–Wasserstein distances and the
        metric approach to object matching. Foundations of computational
        mathematics 11.4 (2011): 417-487.
    .. [38] C. Vincent-Cuaz, T. Vayer, R. Flamary, M. Corneli, N. Courty, Online
        Graph Dictionary Learning, International Conference on Machine Learning
        (ICML), 2021.
    """
    p, q = list_to_array(p, q)

    p0, q0, C10, C20 = p, q, C1, C2
    nx = get_backend(p0, q0, C10, C20)

    p = nx.to_numpy(p)
    q = nx.to_numpy(q)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)
    
    if G0 is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = nx.to_numpy(G0)
        # Check marginals of G0
        np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-04)
        np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-04)

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    T, log_gw = cg(p, q, 0, 1, f, df, G0, log=True, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)

    #T0 = nx.from_numpy(T, type_as=C10)

    #log_gw['gw_dist'] = nx.from_numpy(gwloss(constC, hC1, hC2, T), type_as=C10)
    gp = nx.from_numpy(log_gw['u'] - log_gw['u'].mean())
    gq = nx.from_numpy(log_gw['v'] - log_gw['v'].mean())
    #log_gw['T'] = T0

    if loss_fun == 'square_loss':
        
        gC1 = nx.from_numpy(2 * C1 * (p[:, None] * p[None, :]) - 2 * T.dot(C2).dot(T.T))
        gC2 = nx.from_numpy(2 * C2 * (q[:, None] * q[None, :]) - 2 * T.T.dot(C1).dot(T))
        #gw = nx.set_gradients(gw, (p0, q0, C10, C20),
        #                     (log_gw['u'], log_gw['v'], gC1, gC2))
    #for checking backprop manually
    #return T0, nx.from_numpy(gwloss(constC, hC1, hC2, T), type_as=C10), nx.from_numpy(log_gw['u']), nx.from_numpy(log_gw['v']), gC1, gC2
    return nx.from_numpy(gwloss(constC, hC1, hC2, T), type_as=C10), gp, gq, gC1, gC2


def parallel_fused_gromov_wasserstein2_learnablealpha( C1, C2, F1, F2, M, p, q, loss_fun='square_loss', alpha=0.5, compute_gradients=True, learn_alpha=False, armijo=False, log=False, G0=None, **kwargs):
    r"""
    Computes the FGW distance between two graphs see (see :ref:`[24] <references-fused-gromov-wasserstein2>`)
    .. math::
        \min_\gamma \quad (1 - \alpha) \langle \gamma, \mathbf{M} \rangle_F + \alpha \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}
        s.t. \ \mathbf{\gamma} \mathbf{1} &= \mathbf{p}
             \mathbf{\gamma}^T \mathbf{1} &= \mathbf{q}
             \mathbf{\gamma} &\geq 0
    where :
    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\mathbf{p}` and :math:`\mathbf{q}` are source and target weights (sum to 1)
    - `L` is a loss function to account for the misfit between the similarity matrices
    The algorithm used for solving the problem is conditional gradient as
    discussed in :ref:`[24] <references-fused-gromov-wasserstein2>`
    Note that when using backends, this loss function is differentiable wrt the
    marices and weights for quadratic loss using the gradients from [38]_.
    Parameters
    ----------
    M : array-like, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : array-like, shape (ns, ns)
        Metric cost matrix representative of the structure in the source space.
    C2 : array-like, shape (nt, nt)
        Metric cost matrix representative of the structure in the target space.
    p :  array-like, shape (ns,)
        Distribution in the source space.
    q :  array-like, shape (nt,)
        Distribution in the target space.
    loss_fun : str, optional
        Loss function used for the solver.
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research.
        Else closed form is used. If there are convergence issues use False.
    log : bool, optional
        Record log if True.
    **kwargs : dict
        Parameters can be directly passed to the ot.optim.cg solver.
    Returns
    -------
    fgw-distance : float
        Fused gromov wasserstein distance for the given parameters.
    log : dict
        Log dictionary return only if log==True in parameters.
    .. _references-fused-gromov-wasserstein2:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    .. [38] C. Vincent-Cuaz, T. Vayer, R. Flamary, M. Corneli, N. Courty, Online
        Graph Dictionary Learning, International Conference on Machine Learning
        (ICML), 2021.
    """
    p, q = list_to_array(p, q)

    p0, q0, C10, C20, F10, F20, M0, alpha0 = p, q, C1, C2, F1, F2, M, alpha
    nx = get_backend(p0, q0, C10, C20, F10, F20, M0, alpha0)

    p = nx.to_numpy(p0)
    q = nx.to_numpy(q0)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)
    F1 = nx.to_numpy(F10)
    F2 = nx.to_numpy(F20)
    M = nx.to_numpy(M0)
    alpha = nx.to_numpy(alpha0)
    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    if G0 is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = nx.to_numpy(G0)
        # Check marginals of G0
        np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-04)
        np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-04)

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    T, log_fgw = cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True, **kwargs)

    fgw_dist = nx.from_numpy(log_fgw['loss'][-1], type_as=C10)
    if not compute_gradients:
        return fgw_dist
    else:
        #T0 = nx.from_numpy(T, type_as=C10)
    
        #log_fgw['fgw_dist'] = fgw_dist
        #log_fgw['u'] = nx.from_numpy(log_fgw['u'], type_as=C10)
        #log_fgw['v'] = nx.from_numpy(log_fgw['v'], type_as=C10)
        #log_fgw['T'] = T0
    
        if loss_fun == 'square_loss':
            gC1 = nx.from_numpy(2 * C1 * (p[:, None] * p[None, :]) - 2 * T.dot(C2).dot(T.T))
            gC2 = nx.from_numpy(2 * C2 * (q[:, None] * q[None, :]) - 2 * T.T.dot(C1).dot(T))
            if learn_alpha:
                gwloss_ = gwloss(constC, hC1, hC2, T)
                galpha = nx.from_numpy(gwloss_ - (M*T).sum(), type_as=C10)
            else:
                galpha = None
            #fgw_dist = nx.set_gradients(fgw_dist, (p0, q0, C10, C20, M0, alpha0),
            #                            (log_fgw['u'], log_fgw['v'], alpha0 * gC1, alpha0 * gC2, (1 - alpha0) * T0, galpha))
        gp = nx.from_numpy(log_fgw['u'] - log_fgw['u'].mean())
        gq = nx.from_numpy(log_fgw['v'] - log_fgw['v'].mean())
        gF1 = nx.from_numpy(2 * F1 * p[:, None] - 2 * T.dot(F2))
        gF2 = nx.from_numpy(2 * F2 * q[:, None] - 2 * (T.T).dot(F1))
        
        return fgw_dist, gp, gq, alpha0 * gC1, alpha0 * gC2, (1. - alpha0) * gF1, (1. - alpha0) * gF2, galpha
    
#%%

from torch.autograd import Function
class ValFunction(Function):

    @staticmethod
    def forward(ctx, val, grads, *inputs):
        ctx.grads = grads
        return val

    @staticmethod
    def backward(ctx, grad_output):
        # the gradients are grad
        return (None, None) + tuple(g * grad_output for g in ctx.grads)

    

def set_gradients(Func, val, inputs, grads):

    res = Func.apply(val, grads, *inputs)

    return res

#%%

def probability_simplex_projection(x):
    descending_idx = th.argsort(x, descending=True)
    u = x[descending_idx]
    rho= 0.
    lambda_= 1.
    for i in range(u.shape[0]):
        value = u[i] + (1- u[:(i+1)].sum())/(i+1)
        if value>0:
            rho+=1
            lambda_-=u[i]
        else:
            break
    return th.max(x + lambda_/rho, th.zeros_like(x))

#%% 