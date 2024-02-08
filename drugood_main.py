from dataset.good_hiv import GOODHIV
import os.path as osp
import torch
import math
import time

import logging
from tqdm import tqdm
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader

from models.gnn import GNN
from configures.arguments import load_arguments_from_yaml, get_args
from dataset.get_datasets import get_dataset
from utils import AverageMeter, validate, print_info, init_weights, load_generator, ImbalancedSampler
from utils import build_augmentation_dataset

cls_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
reg_criterion = torch.nn.MSELoss(reduction='none')

import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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


def train(args, model, train_loaders, optimizer, scheduler, epoch):
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
    loss_all = []
    for batch_labeled in train_loaders:
        end = time.time()
        model.zero_grad()

        batch_labeled = batch_labeled.to(device)
        targets = batch_labeled.y.to(torch.float32)
        print(batch_labeled)
        is_labeled = targets == targets
        try:
            pred_labeled = model(batch_labeled)[0]
            print(pred_labeled.size())
        except:
            continue
        Losses = criterion(pred_labeled.view(targets.size()).to(torch.float32), targets)
        loss = Losses.mean()
        loss_all.append(loss)
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
    #loss_mean = torch.tensor(loss_all).mean()
    #print(loss_all)
    return train_loaders,loss_all


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
    import os
    dataset_dir = 'data/'
    datetime_now = datetime.now().strftime("%Y%m%d.%H%M%S")

    exp_dir = osp.join('local/', datetime_now)
    os.makedirs(exp_dir, exist_ok=True)
    from dataset.register import register
    register.datasets ={"GOODHIV": GOODHIV(root=osp.join(dataset_dir, 'hiv/'),
                                                          domain="scaffold",
                                                          shift="covariate")}
    
    dataset, meta_info = register.datasets["GOODHIV"].load(dataset_root=osp.join(dataset_dir, 'hiv/'),
                                                          domain="scaffold",
                                                          shift="covariate",
                                                         )
    iid_dataset = dataset["train"]
    #ood_dataset = DrugOOD(osp.join(dataset_dir, 'DrugOOD/'), mode='ood')
    
    

    n_train_data, n_val_data, n_in_test_data, n_out_test_data = 1024, 512, 512, 512
    train_dataset, in_test_dataset = iid_dataset[:n_train_data], dataset["id_test"][:n_in_test_data]
    
    out_test_dataset = dataset["test"][:n_out_test_data]

    # we need to modify outliers' idx to track their gradients (for GraphDE-v)
    
    
    id_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    ood_test_loader = DataLoader(out_test_dataset, batch_size=args.batch_size, shuffle=False)


    model = GNN(gnn_type=args.model, num_tasks=labeled_dataset.num_tasks, num_layer=args.num_layer,
                emb_dim=args.emb_dim,
                drop_ratio=args.drop_ratio, graph_pooling=args.readout, norm_layer=args.norm_layer).to(device)



    init_weights(model, args.initw_name, init_gain=0.02)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 100)
    logging.warning(f"device: {args.device}, " f"n_gpu: {args.n_gpu}, ")
    logger.info(dict(args._get_kwargs()))
    logger.info("***** Running training *****")
    logger.info(
        f"  Task = {args.dataset}@{args.num_trained}/{len(label_split_idx['valid'])}/{len(label_split_idx['test'])}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Total train batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.epochs * args.steps}")

    train_loss =[]
    test_mae = []
    for epoch in range(0, args.epochs):
        train_loaders,loss_all = train(args, model, id_train_loader, optimizer, scheduler, epoch)
        train_loss.append(all)
        #train_perf = validate(args, model, train_loaders)
        test_perf = validate(args, model, ood_test_loader)
        test_mae.append(test_perf['mae'])

    return train_loss, test_mae


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
    for exp_num in range(args.trails):
        seed_torch(exp_num)
        args.exp_num = exp_num
        
        train_loss, test_mae = main(args)
        print(train_loss)
        print(test_mae)