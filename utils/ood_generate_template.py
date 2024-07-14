import time
import torch
from tqdm import trange
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset

from .sde import VESDE
from .infonce import InfoNCE
from .solver import LangevinCorrector, ReverseDiffusionPredictor, mask_x, mask_adjs, gen_noise, get_score_fn

from .mol_utils import convert_dense_to_rawpyg, convert_sparse_to_dense, combine_graph_inputs, \
    convert_dense_adj_to_sparse_with_attr, convert_to_batch
from .mol_utils import extract_graph_feature, estimate_feature_embs
from utils.FGW import *
import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
__all__ = ['build_augmentation_dataset']

cls_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
reg_criterion = torch.nn.MSELoss(reduction='none')


# -------- get negative samples for infoNCE loss --------
def get_negative_indices(y_true, n_sample=10):
    if torch.isnan(y_true).sum() != 0:
        print('y_true', (y_true == y_true).size(), (y_true == y_true).sum())
        return None
    y_true = torch.nan_to_num(y_true, nan=0.)
    task_num = y_true.size(1)
    diffs = torch.abs(y_true.view(-1, 1, task_num) - y_true.view(1, -1, task_num)).mean(dim=-1)
    diffs_desc_indices = torch.argsort(diffs, dim=1, descending=True)
    return diffs_desc_indices[:, :n_sample]


def inner_sampling(args, generator, x, adj, sde_x, sde_adj, diff_steps, flags=None):
    score_fn_x = get_score_fn(sde_x, generator['model_x'], perturb_ratio=args.perturb_ratio)
    score_fn_adj = get_score_fn(sde_adj, generator['model_adj'], perturb_ratio=args.perturb_ratio)
    snr, scale_eps, n_steps = args.snr, args.scale_eps, args.n_steps
    predictor_obj_x = ReverseDiffusionPredictor('x', sde_x, score_fn_x, False, perturb_ratio=args.perturb_ratio)
    corrector_obj_x = LangevinCorrector('x', sde_x, score_fn_x, snr, scale_eps, n_steps,
                                        perturb_ratio=args.perturb_ratio)
    predictor_obj_adj = ReverseDiffusionPredictor('adj', sde_adj, score_fn_adj, False, perturb_ratio=args.perturb_ratio)
    corrector_obj_adj = LangevinCorrector('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps,
                                          perturb_ratio=args.perturb_ratio)
    x, adj = mask_x(x, flags), mask_adjs(adj, flags)
    total_sample_steps = args.out_steps
    timesteps = torch.linspace(1, 1e-3, total_sample_steps, device=args.device)[-diff_steps:]
    with torch.no_grad():
        # -------- Reverse diffusion process --------
        for i in range(diff_steps):
            t = timesteps[i]
            vec_t = torch.ones(adj.shape[0], device=t.device) * t
            _x = x
            
            x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t)
            adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t)
            _x = x
            x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t)
            adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t)
    return x_mean.detach(), adj_mean.detach()


def get_mask(data_loader):
    num_nodes_list = []

    # 计算每个图的节点数量

    for i in range(data_loader.shape[0]):
        num_nodes = data_loader[i].shape[0]  # 获取每个图的节点数量
        num_nodes_list.append(num_nodes)

    # 计算节点数量的中位数
    median_num_nodes = torch.median(torch.tensor(num_nodes_list)).item()

    # 创建掩码
    mask = []
    tag = False
    for num_nodes in num_nodes_list:
        if num_nodes == median_num_nodes:
            if tag:
                mask.append(False)
            else:
                mask.append(True)
                tag = True
        else:
            mask.append(False)

    mask = torch.tensor(mask)
    return mask

def standardize_adj(adj):
    device = adj.device
    adj = (adj + adj.transpose(-1,-2)) / 2
    mask = torch.eye(adj.size(-1), adj.size(-1)).bool().unsqueeze_(0).to(device)
    adj.masked_fill_(mask, 0)
    return adj

def quantize_mol(adjs):  
    if type(adjs).__name__ == 'Tensor':
        adjs = adjs.detach().cpu()
    else:
        adjs = torch.tensor(adjs)
    torch.nan_to_num(adjs, nan=0.0)
    adjs = standardize_adj(adjs)
    adjs[adjs >= 2.5] = 3
    adjs[torch.bitwise_and(adjs >= 1.5, adjs < 2.5)] = 2
    adjs[torch.bitwise_and(adjs >= 0.5, adjs < 1.5)] = 1
    adjs[adjs < 0.5] = 0
    return adjs

## -------- Main function for augmentation--------##
def build_augmentation_dataset(args, generator, generator_ood, labeled_data):
    if args.dataset.startswith('nx'):
        raise NotImplementedError(f"currently not implemented.")


    kept_pyg_list = []
    augmented_pyg_list = []
    augment_fails = 0

    labeled_trainloader = DataLoader(labeled_data, batch_size=args.aug_batch, shuffle=False, num_workers = 0)
    template_adj = []
    template_feature = []
    template_h = []
    for step, batch_data in enumerate(labeled_trainloader):
        batch_data_list = batch_data.to_data_list()
        
        batch_data = batch_data.to(args.device)
        y_true_all, batch_index = batch_data.y.to(torch.float64), batch_data.batch
        y_true_all = y_true_all.unsqueeze(1)
        if batch_data.x.shape[0] >= 1:
            #print(batch_data.x.shape)
            random_indices = torch.randint(low=0, high=batch_data.y.shape[0], size=[1, ], dtype=torch.int64,
                                           device=args.device)
            random_indices_batch = torch.randint(low=0, high=batch_data.y.shape[0], size=[args.topk, ],
                                                 dtype=torch.int64, device=args.device)
            random_augment_mask = torch.zeros(batch_data.y.shape[0]).to(batch_data.y.device).scatter_(0, random_indices,
                                                                                                      1).bool()

            random_batch_mask = torch.zeros(batch_data.y.shape[0]).to(batch_data.y.device).scatter_(0,
                                                                                                    random_indices_batch,
                                                                                                    1).bool()
            augment_labels = y_true_all[random_augment_mask]

            batch_dense_x, batch_dense_x_disable, batch_dense_enable_adj, batch_dense_disable_adj, batch_node_mask = \
                convert_to_batch(batch_index, batch_data.x, batch_data.edge_index, batch_data.edge_attr,
                                        augment_mask=None)
            random_augment_mask = get_mask(batch_dense_enable_adj)
            

            # sde
            total_sample_steps = args.out_steps
            sde_x = VESDE(sigma_min=0.1, sigma_max=1, N=total_sample_steps)
            sde_adj = VESDE(sigma_min=0.1, sigma_max=1, N=total_sample_steps)

            # batch_dense_x, batch_dense_enable_adj = batch_dense_x[random_augment_mask], batch_dense_enable_adj[random_augment_mask]
            batch_dense_x_select, batch_dense_enable_adj_select = batch_dense_x[random_augment_mask], \
                                                                  batch_dense_enable_adj[random_augment_mask]
            if batch_dense_x_disable is not None:
                batch_dense_x_disable_select, batch_dense_disable_adj_select = batch_dense_x_disable[
                                                                                   random_augment_mask], \
                                                                               batch_dense_disable_adj[
                                                                                   random_augment_mask]

            # perturb x
            #print(batch_dense_x_select.shape)
            peturb_t = torch.ones(batch_dense_enable_adj_select.shape[0]).to(args.device) * (sde_adj.T - 1e-3) + 1e-3
            mean_x, std_x = sde_x.marginal_prob(batch_dense_x_select, peturb_t)
            z_x = gen_noise(batch_dense_x_select.to(torch.float32), flags=batch_node_mask[random_augment_mask],
                            sym=False, perturb_ratio=args.perturb_ratio)
            perturbed_x = mean_x + std_x[:, None, None] * z_x
            perturbed_x = mask_x(perturbed_x, batch_node_mask[random_augment_mask])

            # perturb adj
            mean_adj, std_adj = sde_adj.marginal_prob(batch_dense_enable_adj_select, peturb_t)
            z_adj = gen_noise(batch_dense_enable_adj_select.to(torch.float32),
                              flags=batch_node_mask[random_augment_mask], sym=True, perturb_ratio=args.perturb_ratio)
            perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
            perturbed_adj = mask_adjs(perturbed_adj, batch_node_mask[random_augment_mask])

            timesteps = torch.linspace(1, 1e-3, total_sample_steps, device=args.device)[-args.out_steps:]

            def get_aug_grads(inner_output_data, input_batch_data, input_ood_data):
                ood_x, ood_adj = input_ood_data
                batch_x, batch_adj = input_batch_data
                inner_output_x, inner_output_adj = inner_output_data
                inner_output_adj = mask_adjs(inner_output_adj, batch_node_mask[random_augment_mask])
                inner_output_x = mask_x(inner_output_x, batch_node_mask[random_augment_mask])
                # template
                inner_output_x, inner_output_adj = inner_output_x.requires_grad_(), inner_output_adj.requires_grad_()

                with torch.enable_grad():

                    if inner_output_x.shape[0] >= 1:

                        list_of_tensors = th.split(batch_adj, 1, dim=0)
                        list_of_tensors = [t.squeeze(dim=0).to('cpu') for t in list_of_tensors]
                        list_of_features = th.split(batch_x, 1, dim=0)
                        list_of_features = [t.squeeze(dim=0).to('cpu') for t in list_of_features]
                        list_h = [th.ones(C.shape[0], dtype=th.float64, device='cpu') / C.shape[0] for C in
                                  list_of_tensors]
                        hbar = [
                            th.ones(inner_output_adj.shape[1], dtype=th.float64, device='cpu') / inner_output_adj.shape[
                                1]]
                        adj_tmp = [inner_output_adj.squeeze(dim=0).requires_grad_().to('cpu')]
                        fea_tmp = [inner_output_x.squeeze(dim=0).requires_grad_().to('cpu')]
                      
                        loss_id = parallelized_get_features(1, list_of_tensors, list_of_features, list_h, adj_tmp,
                                                            fea_tmp, hbar, th.tensor(0.5, dtype=torch.float64)).mean()
                   
                        loss_ood = parallelized_get_features(1, [ood_adj.squeeze(dim=0).to('cpu')],
                                                             [ood_x.squeeze(dim=0).to('cpu')], [
                                                                 th.ones(ood_adj.shape[1], dtype=th.float64,
                                                                         device='cpu') / ood_adj.shape[1]], adj_tmp,
                                                             fea_tmp, hbar, th.tensor(0.5, dtype=torch.float64)).mean()
                        total_loss = loss_id - loss_ood
                        aug_grad_x, aug_grad_adj = torch.autograd.grad(total_loss, [inner_output_x, inner_output_adj])
                    else:

                        aug_grad_x, aug_grad_adj = None, None
                return aug_grad_x, aug_grad_adj

            score_fn_x = get_score_fn(sde_x, generator['model_x'], perturb_ratio=args.perturb_ratio)
            score_fn_adj = get_score_fn(sde_adj, generator['model_adj'], perturb_ratio=args.perturb_ratio)
            predictor_obj_x = ReverseDiffusionPredictor('x', sde_x, score_fn_x, False, perturb_ratio=args.perturb_ratio)
            corrector_obj_x = LangevinCorrector('x', sde_x, score_fn_x, args.snr, args.scale_eps, args.n_steps,
                                                perturb_ratio=args.perturb_ratio)
            predictor_obj_adj = ReverseDiffusionPredictor('adj', sde_adj, score_fn_adj, False,
                                                          perturb_ratio=args.perturb_ratio)
            corrector_obj_adj = LangevinCorrector('adj', sde_adj, score_fn_adj, args.snr, args.scale_eps, args.n_steps,
                                                  perturb_ratio=args.perturb_ratio)
            if args.no_print:
                outer_iters = range(args.out_steps)
            else:
                outer_iters = trange(0, (args.out_steps), desc='[Outer Sampling]', position=1, leave=False)
            denoise_adj_ood = generator_ood["model_adj"]
            denoise_x_ood = generator_ood["model_x"]
            random_scalar = torch.randn(1).to(args.device)
            for key, param in denoise_adj_ood.state_dict().items():
                noise = torch.ones_like(
                    param) * random_scalar  # or torch.rand_like(param) * scale - offset
                param.add_(noise)
            for key, param in denoise_x_ood.state_dict().items():
                noise = torch.ones_like(
                    param) * random_scalar  # or torch.rand_like(param) * scale - offset
                param.add_(noise)

            inner_output_ood_x, inner_output_ood_adj = inner_sampling(args, generator_ood, perturbed_x, perturbed_adj,
                                                                      sde_x, sde_adj, args.out_steps,
                                                                      batch_node_mask[random_augment_mask])  # ood 样本

            denoise_adj_ood = generator["model_adj"]
            denoise_x_ood = generator["model_x"]
            generator_ood = {"model_x": denoise_x_ood, "model_adj": denoise_adj_ood}
            for i in outer_iters:
                inner_output_x, inner_output_adj = inner_sampling(args, generator, perturbed_x, perturbed_adj, sde_x,
                                                                  sde_adj, args.out_steps - i,
                                                                  batch_node_mask[random_augment_mask])
                aug_grad_x, aug_grad_adj = get_aug_grads([inner_output_x, inner_output_adj],
                                                         [batch_dense_x[random_batch_mask],
                                                          batch_dense_enable_adj[random_batch_mask]],
                                                         [inner_output_ood_x, inner_output_ood_adj])
                with torch.no_grad():
                    t = timesteps[i]
                    vec_t = torch.ones(perturbed_adj.shape[0], device=t.device) * t
                    _x = perturbed_x
                    perturbed_x, perturbed_x_mean = corrector_obj_x.update_fn(perturbed_x, perturbed_adj,
                                                                              batch_node_mask[random_augment_mask],
                                                                              vec_t, aug_grad=aug_grad_x)
                    perturbed_adj, perturbed_adj_mean = corrector_obj_adj.update_fn(_x, perturbed_adj, batch_node_mask[
                        random_augment_mask], vec_t, aug_grad=aug_grad_adj)
                    _x = perturbed_x
                    perturbed_x, perturbed_x_mean = predictor_obj_x.update_fn(perturbed_x, perturbed_adj,
                                                                              batch_node_mask[random_augment_mask],
                                                                              vec_t, aug_grad=aug_grad_x)
                    perturbed_adj, perturbed_adj_mean = predictor_obj_adj.update_fn(_x, perturbed_adj, batch_node_mask[
                        random_augment_mask], vec_t, aug_grad=aug_grad_adj)

            perturbed_adj_mean = mask_adjs(perturbed_adj_mean, batch_node_mask[random_augment_mask])
            perturbed_x_mean = mask_x(perturbed_x_mean, batch_node_mask[random_augment_mask])

            augmented_x, augmented_adj = perturbed_x_mean.cpu(), perturbed_adj_mean.cpu()
            
            augmented_x, augmented_adj = combine_graph_inputs(augmented_x, batch_dense_x_disable[random_augment_mask], augmented_adj, batch_dense_disable_adj[random_augment_mask], mode='discrete')
            
            batch_augment_pyg_list = convert_dense_to_rawpyg(augmented_x, augmented_adj, augment_labels,
                                                             n_jobs=1)
            
            augment_indices = random_augment_mask.nonzero().view(-1).cpu().tolist()
            augmented_pyg_list_temp = []
            for pyg_data in batch_augment_pyg_list:
               
                if not isinstance(pyg_data, int):
                    
                    augmented_pyg_list_temp.append(pyg_data)
                elif args.strategy.split('_')[0] == 'add':
                    pass
                else:
                
                    augment_fails += 1
                    #kept_pyg_list.append(batch_data_list[augment_indices[pyg_data]])

            augmented_pyg_list.extend(augmented_pyg_list_temp)

    kept_pyg_list.extend(augmented_pyg_list)

    new_dataset = NewDataset(kept_pyg_list, num_fail=augment_fails)
 
    return new_dataset, len(new_dataset)


class NewDataset(InMemoryDataset):
    def __init__(self, data_list, num_fail=0, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.data_list = data_list
        self.data_len = len(data_list)
        self.num_fail = num_fail
        # print('data_len', self.data_len, 'num_fail', num_fail)
        self.data, self.slices = self.collate(data_list)

    def get_idx_split(self):
        return {'train': torch.arange(self.data_len, dtype=torch.long), 'valid': None, 'test': None}


if __name__ == '__main__':
    pass