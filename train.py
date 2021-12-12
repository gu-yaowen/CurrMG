import numpy as np
import torch
import os
import pandas as pd
import torch.nn as nn
from dgllife.utils import Meter
from torch.optim import Adam
from utils import load_model, predict
from load_data import load_data_from_dgl, cal_diff_feat, load_data
from utils import init_featurizer, split_dataset, \
    plot_train_method, plot_result, set_seed, \
    criterion, config_update
import time
from train_sampler import diff_metric_get
from model_config import set_model_config


def train_iteration_Curr(args, model, train_data_loader, val_data_loader,
                         loss_criterion, optimizer):
    model.train()
    best_model = model
    best_score = 0 if args['metric'] in ['roc_auc_score', 'pr_auc_score', 'r2'] else 999
    loss_list, val_list = [], []
    time_list = []
    time_list.append(time.time())
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
        time_list.append(time.time())
        model.train()
        loss_list.append(loss.cpu().detach().numpy())
        val_list.append(val_score)
    np.save(os.path.join(args['result_path'], 'running time.npy'),
            np.array(time_list))
    return best_model, best_score, loss_list, val_list


def train_iteration_noCurr(args, model, train_data_loader, val_data_loader,
                           loss_criterion, optimizer):
    model.train()
    best_model = model
    best_score = 0 if args['metric'] in ['roc_auc_score', 'pr_auc_score', 'r2'] else 999
    iter_conut = 0
    loss_list, val_list = [], []
    time_list = []
    time_list.append(time.time())
    for i in range(999):
        if iter_conut == args['t_total']:
            break
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
            time_list.append(time.time())
            model.train()
            loss_list.append(loss.cpu().detach().numpy())
            val_list.append(val_score)
            if iter_conut == args['t_total']:
                break
    np.save(os.path.join(args['result_path'], 'running time.npy'),
            np.array(time_list))
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
            if len(prediction) == 1:
                predict_all.append(prediction.cpu().numpy()[0])
            else:
                predict_all.extend(prediction.cpu().numpy().squeeze().tolist())
            eval_meter.update(prediction, labels, masks)

    return np.mean(eval_meter.compute_metric(args['metric'])), predict_all


def train(args):
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:' + args['cuda_id'].split(' ')[0])
    else:
        args['device'] = torch.device('cpu')
    args['result_path'] = os.path.join(os.getcwd(), args['result_path'])
    try:
        os.mkdir(args['result_path'])
    except:
        pass
    set_seed(args)
    args = init_featurizer(args)
    model_config = set_model_config(args)
    args = config_update(args, model_config)
    if args['featurizer_type'] != 'pre_train':
        args['in_node_feats'] = args['node_featurizer'].feat_size()
        if args['edge_featurizer'] is not None:
            args['in_edge_feats'] = args['edge_featurizer'].feat_size()
    dataset = load_data_from_dgl(args)
    args['n_tasks'] = dataset.n_tasks
    train_set, val_set, test_set = split_dataset(args, dataset)
    args['t_total'] = int(args['num_epochs'] * len(train_set) / args['batch_size'])
    diff_feat = cal_diff_feat(args, dataset, train_set)
    if not args['is_Curr'] and args['threshold'] != 1.0:
        _, diff = diff_metric_get(args, diff_feat)
        if args['diff_type'] == 'Two_Stage':
            diff = diff[0]
        train_idx = np.where(diff < args['threshold'])[0]
        train_set = tuple(np.array(list(train_set))
                          [train_idx].tolist())
    print('Total Iterations: ', args['t_total'])
    train_loader, val_loader, test_loader = load_data(args, train_set,
                                                      val_set, test_set, diff_feat)

    cuda_id = [int(i) for i in args['cuda_id'].split(' ')]
    if (len(cuda_id) > 1) and (torch.cuda.device_count() > 0):
        model = nn.DataParallel(load_model(args),
                                device_ids=cuda_id)
    else:
        model = load_model(args)
    model.to(args['device'])

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

    plot_train_method(args, loss_list, val_list)
    test_score, test_result = eval_iteration(args, best_model, test_loader)
    print('-' * 20 + '+' * 20 + '-' * 20)
    print('val {} {:.4f}'.format(args['metric'], best_score))
    print('test {} {:.4f}'.format(args['metric'], test_score))
    test_result = np.array(test_result).reshape(len(test_set.indices), args['n_tasks'])
    label = dataset.labels.numpy().squeeze()[test_set.indices].reshape(len(test_set.indices), args['n_tasks'])
    smiles = np.array(dataset.smiles)[test_set.indices].reshape(len(test_set.indices), 1)
    data = np.hstack((smiles, label, test_result))
    df = pd.DataFrame(data,
                      columns=['SMILES'] +
                              (['Label_' + str(i) for i in range(args['n_tasks'])] if args['n_tasks'] != 1 else [
                                  'Label']) +
                              (['PREDICT_' + str(i) for i in range(args['n_tasks'])] if args['n_tasks'] != 1 else [
                                  'PREDICT']))
    df.to_csv(os.path.join(args['result_path'], 'result.csv'), index=False)
    if args['n_tasks'] == 1:
        plot_result(args, label, test_result, test_score)
    df = pd.DataFrame(np.array([loss_list, val_list]).T,
                      columns=['Train Loss', 'Validation Score'])
    df.to_csv(os.path.join(args['result_path'], 'loss.csv'), index=False)
    with open(os.path.join(args['result_path'], '{:.4f}'.format(test_score) + '.txt'), 'w') as file:
        file.write(str(test_score))
    return
