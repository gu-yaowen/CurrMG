import numpy as np
import torch
import torch.nn as nn

from dgllife.model import load_pretrained
from dgllife.utils import smiles_to_bigraph, EarlyStopping, Meter
from functools import partial
from torch.optim import Adam
from torch.utils.data import DataLoader
from train_sampler import CurrSampler, CurrBatchSampler, Feat_Calculate
from utils import collate_molgraphs, load_model, predict
from load_data import load_data_from_dgl, cal_diff_feat
from utils import init_featurizer, mkdir_p, split_dataset


def load_data(args, train_set, val_set, test_set,
              diff_feat: np.array):
    if args['is_Curr']:
        print('Training Method in Curriculum Learning')
        sampler = CurrSampler(diff_feat)
        batch_sampler = CurrBatchSampler(sampler, args['batch_size'],
                                         args['t_total'], args['c_type'], random_seed,
                                         args['sample_type'])
        train_loader = DataLoader(train_set,
                                  batch_sampler=batch_sampler,
                                  num_workers=args['num_workers'],
                                  collate_fn=collate_molgraphs)
    else:
        print('Training Method NOT in Curriculum Learning')
        train_loader = DataLoader(train_set, batch_size=args['batch_size'],
                                  num_workers=args['num_workers'],
                                  shuffle=True)
    val_loader = DataLoader(val_set,
                            batch_size=int(len(val_set) * 0.1) if int(len(val_set) * 0.1) < 1000 else 1000,
                            num_workers=args['num_workers'], collate_fn=collate_molgraphs)
    test_loader = DataLoader(test_set,
                             batch_size=int(len(val_set) * 0.1) if int(len(val_set) * 0.1) < 1000 else 1000,
                             num_workers=args['num_workers'], collate_fn=collate_molgraphs)
    return train_loader, val_loader, test_loader


def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        if len(smiles) == 1:
            # Avoid potential issues with batch normalization
            continue

        labels, masks = labels.to(args['device']), masks.to(args['device'])
        prediction = predict(args, model, bg)
        # Mask non-existing labels
        loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels, masks)
        if batch_id % args['print_every'] == 0:
            print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
                epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
    train_score = np.mean(train_meter.compute_metric(args['metric']))
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric'], train_score))


def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args['device'])
            prediction = predict(args, model, bg)
            eval_meter.update(prediction, labels, masks)
    return np.mean(eval_meter.compute_metric(args['metric']))


def train(args):
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:' + args['cuda_id'])
    else:
        args['device'] = torch.device('cpu')

    if args['featurizer_type'] != 'pre_train':
        exp_config['in_node_feats'] = args['node_featurizer'].feat_size()
        if args['edge_featurizer'] is not None:
            exp_config['in_edge_feats'] = args['edge_featurizer'].feat_size()
    exp_config.update({
        'n_tasks': args['n_tasks'],
        'model': args['model']
    })

    dataset = load_data_from_dgl(args)
    train_set, val_set, test_set = split_dataset(args, dataset)
    diff_feat = cal_diff_feat(args, train_set)
    args['t_total'] = int(100 * len(train_set) / args['batch_size'])
    train_loader, val_loader, test_loader = load_data(args, train_set, val_set, test_set, diff_feat)
    return
