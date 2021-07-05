import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
from dgllife.utils import Meter
from torch.optim import Adam
from torch.utils.data import DataLoader
from train_sampler import CurrSampler, CurrBatchSampler
from utils import collate_molgraphs, load_model, predict
from load_data import load_data_from_dgl, cal_diff_feat
from utils import init_featurizer, split_dataset, plot_train_method, plot_result
from model_config import set_model_config


def criterion(args):
    if args['mode'] == 'classification':
        return nn.BCEWithLogitsLoss(reduction='none')
    elif args['mode'] == 'regression':
        return nn.SmoothL1Loss(reduction='none')


def load_data(args, train_set, val_set, test_set,
              diff_feat: np.array):
    if args['is_Curr']:
        print('Training Method in Curriculum Learning')
        sampler = CurrSampler(diff_feat)
        batch_sampler = CurrBatchSampler(sampler, args['batch_size'],
                                         args['t_total'], args['c_type'], args['seed'],
                                         args['sample_type'])
        train_loader = DataLoader(train_set,
                                  batch_sampler=batch_sampler,
                                  num_workers=args['num_workers'],
                                  collate_fn=collate_molgraphs)
    else:
        print('Training Method NOT in Curriculum Learning')
        train_loader = DataLoader(train_set, batch_size=args['batch_size'],
                                  num_workers=args['num_workers'],
                                  shuffle=True, collate_fn=collate_molgraphs)
    val_loader = DataLoader(val_set,
                            batch_size=int(len(val_set) * 0.1) if int(len(val_set) * 0.1) < 1000 else 1000,
                            num_workers=args['num_workers'], collate_fn=collate_molgraphs)
    test_loader = DataLoader(test_set,
                             batch_size=int(len(val_set) * 0.1) if int(len(val_set) * 0.1) < 1000 else 1000,
                             num_workers=args['num_workers'], collate_fn=collate_molgraphs)
    return train_loader, val_loader, test_loader


def train_iteration_Curr(args, model, train_data_loader, val_data_loader,
                         loss_criterion, optimizer):
    model.train()
    best_model = model
    best_score = 0 if args['metric'] in ['roc_auc_score', 'pr_auc_score', 'r2'] else 999
    loss_list, val_list = [], []
    for batch_id, batch_data in enumerate(train_data_loader):
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
        val_score, _ = eval_iteration(args, model, val_data_loader)
        if batch_id % args['print_every'] == 0:
            print('iteration {:d}/{:d}, loss {:.4f}, val_score {:.4f}'.format(
                batch_id, args['t_total'], loss.item(), val_score))
        if args['metric'] in ['roc_auc_score', 'pr_auc_score', 'r2']:
            if val_score > best_score:
                best_model = model
                best_score = val_score
        else:
            if val_score < best_score:
                best_model = model
                best_score = val_score
        loss_list.append(loss.detach().numpy())
        val_list.append(val_score)
    return best_model, best_score, loss_list, val_list


def train_iteration_noCurr(args, model, train_data_loader, val_data_loader,
                           loss_criterion, optimizer):
    model.train()
    best_model = model
    best_score = 0 if args['metric'] in ['roc_auc_score', 'pr_auc_score', 'r2'] else 999
    iter_conut = 0
    loss_list, val_list = [], []
    for i in range(999):
        for batch_id, batch_data in enumerate(train_data_loader):
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
            val_score, _ = eval_iteration(args, model, val_data_loader)
            if iter_conut % args['print_every'] == 0:
                print('iteration {:d}/{:d}, loss {:.4f}, val_score {:.4f}'.format(
                    iter_conut, args['t_total'], loss.item(), val_score))
            if args['metric'] in ['roc_auc_score', 'pr_auc_score', 'r2']:
                if val_score > best_score:
                    best_model = model
                    best_score = val_score
            else:
                if val_score < best_score:
                    best_model = model
                    best_score = val_score
            iter_conut += 1
            loss_list.append(loss.detach().numpy())
            val_list.append(val_score)
        if iter_conut == args['t_total']:
            break
    return best_model, best_score, loss_list, val_list


def eval_iteration(args, model, data_loader):
    predict_all = []
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args['device'])
            prediction = predict(args, model, bg)
            predict_all.extend(prediction.numpy().tolist())
            eval_meter.update(prediction, labels, masks)
    return np.mean(eval_meter.compute_metric(args['metric'])), predict_all


def train(args):
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:' + args['cuda_id'])
    else:
        args['device'] = torch.device('cpu')
    args['result_path'] = os.path.join(os.getcwd(), args['result_path'])
    try:
        os.mkdir(args['result_path'])
    except:
        pass
    args = init_featurizer(args)
    model_config = set_model_config(args)
    args.update(model_config)
    if args['featurizer_type'] != 'pre_train':
        args['in_node_feats'] = args['node_featurizer'].feat_size()
        if args['edge_featurizer'] is not None:
            args['in_edge_feats'] = args['edge_featurizer'].feat_size()
    dataset = load_data_from_dgl(args)
    args['n_tasks'] = dataset.n_tasks
    train_set, val_set, test_set = split_dataset(args, dataset)
    train_smiles = np.array(dataset.smiles)[train_set.indices]
    train_labels = dataset.labels.numpy().squeeze()[train_set.indices]
    if args['n_tasks'] == 1:
        diff_feat = cal_diff_feat(args, train_smiles, train_labels)
    args['t_total'] = int(100 * len(train_set) / args['batch_size'])
    train_loader, val_loader, test_loader = load_data(args, train_set,
                                                      val_set, test_set, diff_feat)
    model = load_model(args).to(args['device'])
    print('Task Type: ', args['mode'])
    loss = criterion(args)
    print('Loss Function:', loss)
    optimizer = Adam(model.parameters(), lr=args['lr'],
                     weight_decay=model_config['weight_decay'])

    if args['is_Curr']:
        best_model, best_score, \
        loss_list, val_list = train_iteration_Curr(args, model, train_loader,
                                                   val_loader, loss, optimizer)
    else:
        best_model, best_score, \
        loss_list, val_list = train_iteration_noCurr(args, model, train_loader,
                                                     val_loader, loss, optimizer)

    plot_train_method(args, loss_list, val_list, best_score)
    test_score, test_result = eval_iteration(args, best_model, test_loader)
    print('-' * 20 + '+' * 20 + '-' * 20)
    print('val {} {:.4f}'.format(args['metric'], best_score))
    print('test {} {:.4f}'.format(args['metric'], test_score))
    label = dataset.labels.numpy().squeeze()[test_set.indices]
    smiles = np.array(dataset.smiles)[test_set.indices]
    df = pd.DataFrame(np.array([smiles, label, test_result]).T,
                      columns=['SMILES', 'LABEL', 'PREDICT'])
    df.to_csv(os.path.join(args['result_path'], 'result.csv'), index=False)
    plot_result(args, label, test_result, test_score)
    return
