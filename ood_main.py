import math
import time
from copy import deepcopy
import logging
from tqdm import tqdm
from datetime import datetime
from utils.FGW import *
from utils.mol_utils import *
import os
import os.path as osp
import numpy as np
import torch
import sklearn.metrics as sk
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
from dataset.drugood_dataset import DrugOOD
from models.gnn import GNN
from configures.arguments import load_arguments_from_yaml, get_args
from dataset.get_datasets import get_dataset
from utils import AverageMeter, validate, print_info, init_weights, load_generator, ImbalancedSampler
from utils.ood_generate_template import build_augmentation_dataset
from utils.FGW import *
cls_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
reg_criterion = torch.nn.MSELoss(reduction='none')
from utils.ood_generate_template import NewDataset
import matplotlib.pyplot as plt
import seaborn as sns
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from utils.naive_generate import build_augmentation_dataset

import torch

import torch.nn.functional as F
from ogb.graphproppred import Evaluator


recall_level_default = 0.95
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def calc_rocauc(loader, predictor):
    evaluator = Evaluator('ogbg-molhiv')
    y_true, y_pred = None, None
    for graph in loader:
        graph.to(device)
        pred_out = predictor(x=graph.x, edge_index=graph.edge_index,
                             edge_attr=graph.edge_attr, batch=graph.batch)
        pred_score = F.softmax(pred_out, dim=1)[:, 1].unsqueeze(1)
        graph.y = graph.y.unsqueeze(1)
        if y_true == None:
            y_true = graph.y
            y_pred = pred_score
        else:
            y_true = torch.cat([y_true, graph.y], dim=0)
            y_pred = torch.cat([y_pred, pred_score], dim=0)
    input_dict = {'y_true':y_true, 'y_pred':y_pred}
    rocauc = evaluator.eval(input_dict)['rocauc']
    return rocauc
def get_logger(name, logfile=None):
    """ create a nice logger """
    logger = logging.getLogger(name)
    # clear handlers if they were created in other runs
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # create console handler add add to logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create file handler add add to logger when name is not None
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def seed_torch(seed=0):
    print('Seed', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=0.5,
                                    # num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(1e-2, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out
def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))

def get_measures_new(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.ones(len(examples), dtype=np.int32)
    labels[:len(pos)] = 0

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr
def plot_distribution(vi_pos, vi_neg, filename):
    """plot vi score distribution figure"""
    sns.set(style='white')
    fig,ax = plt.subplots(1,1,figsize=(4.5,3))

    # plot vi score distribution without ground truth
    sns.distplot(vi_pos, hist=False, ax=ax, kde_kws={'fill': True}, color='#7F95D1', label='in-distribution')
    sns.distplot(vi_neg, hist=False, ax=ax, kde_kws={'fill': True}, color='#FF82A9', label='out-of-distribution')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.set_xlabel('VI Probability Score w/o Ground Truth', size=15)
    ax.set_ylabel('Frequency', size=15)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=10,ncol=2, bbox_to_anchor=(0.55, 1.08))
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
def ood_detection(iid_loader, ood_loader, template_loader,exp_dir,seed):
    pos_prob_scores_max, pos_prob_scores, neg_prob_scores_max, neg_prob_scores = [], [], [], []
    pos_labels, pos_ids, neg_labels, neg_ids = [], [], [], []
    template = next(iter(template_loader))
    template_dense_x, template_dense_x_disable, template_dense_enable_adj, template_dense_disable_adj, template_node_mask = \
        convert_to_batch(template.batch, template.x, template.edge_index, template.edge_attr,
                                augment_mask=None)
    temlpate_of_tensors = th.split(template_dense_enable_adj, 1, dim=0)
    temlpate_of_tensors = [t.squeeze(dim=0).to('cpu') for t in temlpate_of_tensors]
    temlpate_of_features = th.split(template_dense_x, 1, dim=0)
    temlpate_of_features = [t.squeeze(dim=0).to('cpu') for t in temlpate_of_features]
    
    temlpate_h = [th.ones(C.shape[0], dtype=th.float64, device='cpu') / C.shape[0] for C in
              temlpate_of_tensors]
    for graph in iid_loader:
        graph.to(device)
        graph_dense_x, graph_dense_x_disable, graph_dense_enable_adj, graph_dense_disable_adj, graph_node_mask = \
            convert_to_batch(graph.batch, graph.x, graph.edge_index, graph.edge_attr,
                                    augment_mask=None)
        list_of_tensors = th.split(graph_dense_enable_adj, 1, dim=0)
        list_of_tensors = [t.squeeze(dim=0).to('cpu') for t in list_of_tensors]
        list_of_features = th.split(graph_dense_x, 1, dim=0)
        list_of_features = [t.squeeze(dim=0).to('cpu') for t in list_of_features]
        list_h = [th.ones(C.shape[0], dtype=th.float64, device='cpu') / C.shape[0] for C in
                  list_of_tensors]
        
        pos_prob_score = parallelized_get_features(len(temlpate_of_features), list_of_tensors, list_of_features,
                                                    list_h, temlpate_of_tensors, temlpate_of_features, temlpate_h,
                                                    th.tensor(0.5))
        #pos_prob_score_s, _ = torch.max(F.softmax(pos_prob_score, dim=1), dim=1)
        #pos_prob_score_s = 1.0/pos_prob_score_s
        pos_prob_score_s, _ = torch.max(F.softmax(pos_prob_score, dim=1), dim=1)
        pos_prob_score_s = 1.0/pos_prob_score_s
        pos_prob_score_max, _ = torch.median(pos_prob_score, dim=1)
        pos_prob_score_max = 1.0/pos_prob_score_max
        try:
            pos_prob_scores =pos_prob_scores+ pos_prob_score_s.detach().cpu().numpy().tolist()
            pos_prob_scores_max = pos_prob_scores_max + pos_prob_score_max.detach().cpu().numpy().tolist()
        except:
            pos_prob_scores.extend(pos_prob_scores)
    for graph in ood_loader:
        graph.to(device)
        graph_dense_x, graph_dense_x_disable, graph_dense_enable_adj, graph_dense_disable_adj, graph_node_mask = \
            convert_to_batch(graph.batch, graph.x, graph.edge_index, graph.edge_attr,
                                    augment_mask=None)
        list_of_tensors = th.split(graph_dense_enable_adj, 1, dim=0)
        list_of_tensors = [t.squeeze(dim=0).to('cpu') for t in list_of_tensors]
        list_of_features = th.split(graph_dense_x, 1, dim=0)
        list_of_features = [t.squeeze(dim=0).to('cpu') for t in list_of_features]
        list_h = [th.ones(C.shape[0], dtype=th.float64, device='cpu') / C.shape[0] for C in
                  list_of_tensors]

        neg_prob_score = parallelized_get_features(len(temlpate_of_features), list_of_tensors, list_of_features,
                                                    list_h, temlpate_of_tensors, temlpate_of_features, temlpate_h,
                                                    th.tensor(0.5))
        neg_prob_score_s, _ = torch.max(F.softmax(neg_prob_score, dim=1), dim=1)
        neg_prob_score_s = 1.0/neg_prob_score_s
        neg_prob_score_max, _ = torch.median(neg_prob_score, dim=1)
        neg_prob_score_max = 1.0/neg_prob_score_max
        try:
        
            neg_prob_scores =neg_prob_scores+ neg_prob_score_s.detach().cpu().numpy().tolist()
            neg_prob_scores_max = neg_prob_scores_max + neg_prob_score_max.detach().cpu().numpy().tolist()
        except:
            neg_prob_scores.extend(neg_prob_score.detach().cpu().numpy().tolist())

    base_auroc, base_aupr, base_fpr = get_measures(pos_prob_scores_max, neg_prob_scores_max)
    vi_auroc, vi_aupr, vi_fpr = get_measures_new(pos_prob_scores, neg_prob_scores)
    # plot_distribution_comparison(pos_softmax_scores, neg_softmax_scores, pos_prob_scores, neg_prob_scores, os.path.join(exp_dir,f"distribution-seed{seed}.pdf"))
    #plot_distribution(pos_prob_scores, neg_prob_scores, os.path.join(exp_dir,f"distribution-seed{seed}.pdf"))
    plot_distribution(pos_prob_scores, neg_prob_scores, os.path.join(exp_dir,f"distribution-seed{seed}.pdf"))
    plot_distribution(pos_prob_scores_max, neg_prob_scores_max, os.path.join(exp_dir,f"distributionmax-seed{seed}.pdf"))
    print(pos_prob_scores)
    print(neg_prob_scores)
    print(pos_prob_scores_max)
    print(neg_prob_scores_max)
    return base_auroc, base_aupr, base_fpr, vi_auroc, vi_aupr, vi_fpr
def train(args, model, train_loaders, optimizer, scheduler, epoch,template_graph):
    if args.task_type in 'regression':
        criterion = reg_criterion
    else:
        criterion = cls_criterion
    if not args.no_print:
        p_bar = tqdm(range(args.steps))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    device = args.device
    model.train()
    for batch_idx in range(args.steps):
        end = time.time()
        model.zero_grad()
        try:
            batch_labeled = next(train_loaders['labeled_iter'])
        except:
            train_loaders['labeled_iter'] = iter(train_loaders['labeled_trainloader'])
            batch_labeled = next(train_loaders['labeled_iter'])
        batch_labeled = batch_labeled.to(device)
        targets = batch_labeled.y.to(torch.float32)
        is_labeled = targets == targets
        if batch_labeled.x.shape[0] == 1 or batch_labeled.batch[-1] == 0:
            continue
        else:
            pred_labeled = model(batch_labeled,template_graph)[0]
            Losses = criterion(pred_labeled.view(targets.size()).to(torch.float32)[is_labeled], targets[is_labeled])
            loss = Losses.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        if not args.no_print:
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.8f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.steps,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                ))
            p_bar.update()
    if not args.no_print:
        p_bar.close()
    return train_loaders


def main(args):
    device = torch.device('cuda', args.gpu_id)
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    labeled_dataset = get_dataset(args, './raw_data')
    label_split_idx = labeled_dataset.get_idx_split()
    args.num_trained = len(label_split_idx["train"])
    args.num_trained_init = args.num_trained
    args.task_type = labeled_dataset.task_type
    args.steps = args.num_trained // args.batch_size + 1
    args.strategy = args.strategy_init

    
    dataset_dir = 'data/'
    datetime_now = datetime.now().strftime("%Y%m%d.%H%M%S")
    
    exp_dir = osp.join('local/', datetime_now)
    os.makedirs(exp_dir, exist_ok=True)
    iid_dataset = DrugOOD(osp.join(dataset_dir, 'DrugOOD/'), mode='iid')

    ood_dataset = DrugOOD(osp.join(dataset_dir, 'DrugOOD/'), mode='ood')

    perm_idx = torch.randperm(len(iid_dataset), generator=torch.Generator().manual_seed(0))
    iid_dataset = iid_dataset[perm_idx]


    perm_idx = torch.randperm(len(ood_dataset), generator=torch.Generator().manual_seed(0))
    ood_dataset = ood_dataset[perm_idx]

    n_train_data, n_val_data, n_in_test_data, n_out_test_data = 1000, 500, 500, 500
    train_dataset, in_test_dataset = iid_dataset[:n_train_data], iid_dataset[n_train_data:n_train_data + n_in_test_data]
    val_dataset = iid_dataset[-n_val_data:]
    out_test_dataset = ood_dataset[:n_out_test_data]



    # we need to modify outliers' idx to track their gradients (for GraphDE-v)
    for i, data in enumerate(train_dataset):
        
        data.idx += (i - data.idx)


    id_test_loader = DataLoader(in_test_dataset, batch_size=args.batch_size, shuffle=False)
    ood_test_loader = DataLoader(out_test_dataset, batch_size=args.batch_size, shuffle=False)

    generator = load_generator(device, path='checkpoints/qm9_denoise.pth')
    generator_ood = load_generator(device, path='checkpoints/qm9_denoise.pth')
    template_dataset, template_numbers = build_augmentation_dataset(args, generator, generator_ood, train_dataset)

    tempate_dataloader = DataLoader(template_dataset, batch_size=template_numbers, shuffle=False,
                                    num_workers=args.num_workers)
    for batch_data in id_test_loader:
        batch_data = batch_data.to(args.device)
        y_true_all, batch_index = batch_data.y.to(torch.float64), batch_data.batch
        y_true_all = y_true_all.unsqueeze(1)
        batch_dense_x, batch_dense_x_disable, batch_dense_enable_adj, batch_dense_disable_adj, batch_node_mask = \
            convert_to_batch(batch_index, batch_data.x, batch_data.edge_index, batch_data.edge_attr,
                                    augment_mask=None)
        augmented_x, augmented_adj = combine_graph_inputs(batch_dense_x, batch_dense_x_disable,
                                                          batch_dense_enable_adj, batch_dense_disable_adj,
                                                          mode='discrete')
        batch_augment_pyg_list = convert_dense_to_rawpyg(augmented_x, augmented_adj, y_true_all,
                                                         n_jobs=1)
        in_test_dataset = NewDataset(batch_augment_pyg_list, num_fail=len(in_test_dataset))
    for batch_data in ood_test_loader:
        batch_data = batch_data.to(args.device)
        y_true_all, batch_index = batch_data.y.to(torch.float64), batch_data.batch
        y_true_all = y_true_all.unsqueeze(1)
        batch_dense_x, batch_dense_x_disable, batch_dense_enable_adj, batch_dense_disable_adj, batch_node_mask = \
            convert_to_batch(batch_index, batch_data.x, batch_data.edge_index, batch_data.edge_attr,
                                    augment_mask=None)
        augmented_x, augmented_adj = combine_graph_inputs(batch_dense_x.to(device), batch_dense_x_disable.to(device),
                                                          batch_dense_enable_adj.to(device), batch_dense_disable_adj.to(device),
                                                          mode='discrete')
        batch_augment_pyg_list = convert_dense_to_rawpyg(augmented_x.to(device), augmented_adj.to(device), y_true_all.to(device),
                                                         n_jobs=1)
        out_test_dataset = NewDataset(batch_augment_pyg_list, num_fail=len(out_test_dataset))
    id_test_loader = DataLoader(in_test_dataset, batch_size=16, shuffle=False)
    ood_test_loader = DataLoader(out_test_dataset, batch_size=16, shuffle=False)
    base_auroc, base_aupr, base_fpr, vi_auroc, vi_aupr, vi_fpr = ood_detection(id_test_loader, ood_test_loader, tempate_dataloader,exp_dir,46)
    print(base_auroc)
    print(base_aupr)
    print(base_fpr)
    print(vi_auroc)
    print(vi_aupr)
    print(vi_fpr)



if __name__ == '__main__':
    args = get_args()

    config = load_arguments_from_yaml(f'configures/{args.dataset}.yaml')
    for arg, value in config.items():
        setattr(args, arg, value)

    args.strategy_init = args.strategy
    datetime_now = datetime.now().strftime("%Y%m%d.%H%M%S")
    logger = get_logger(__name__, logfile=None)
    print(args)
    results = {}
    exp_dir = osp.join('local/', datetime_now)
    os.makedirs(exp_dir, exist_ok=True)
        
    main(args)
    
        