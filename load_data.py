import os
import numpy as np
import pandas as pd
from dgllife.data import FreeSolv, Lipophilicity, ESOL, \
    BBBP, BACE, ClinTox, HIV, Tox21
from functools import partial
from dgllife.utils import smiles_to_bigraph
from train_sampler import Feat_Calculate
from dgllife.data.csv_dataset import MoleculeCSVDataset


def load_data_from_dgl(args):
    if args['dataset'] == 'FreeSolv':
        dataset = FreeSolv(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                           node_featurizer=args['node_featurizer'],
                           edge_featurizer=args['edge_featurizer'],
                           n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        args['mode'] = 'regression'
    elif args['dataset'] == 'Lipophilicity':
        dataset = Lipophilicity(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                                node_featurizer=args['node_featurizer'],
                                edge_featurizer=args['edge_featurizer'],
                                n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        args['mode'] = 'regression'
    elif args['dataset'] == 'ESOL':
        dataset = ESOL(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                       node_featurizer=args['node_featurizer'],
                       edge_featurizer=args['edge_featurizer'],
                       n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        args['mode'] = 'regression'
    elif args['dataset'] == 'Tox21':
        dataset = Tox21(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                        node_featurizer=args['node_featurizer'],
                        edge_featurizer=args['edge_featurizer'],
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        args['mode'] = 'classification'
    elif args['dataset'] == 'HIV':
        dataset = HIV(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                      node_featurizer=args['node_featurizer'],
                      edge_featurizer=args['edge_featurizer'],
                      n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        args['mode'] = 'classification'
    elif args['dataset'] == 'BBBP':
        dataset = BBBP(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                       node_featurizer=args['node_featurizer'],
                       edge_featurizer=args['edge_featurizer'],
                       n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        args['mode'] = 'classification'
    elif args['dataset'] == 'BACE':
        dataset = BACE(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                       node_featurizer=args['node_featurizer'],
                       edge_featurizer=args['edge_featurizer'],
                       n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        args['mode'] = 'classification'
    elif args['dataset'] == 'ClinTox':
        dataset = ClinTox(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                          node_featurizer=args['node_featurizer'],
                          edge_featurizer=args['edge_featurizer'],
                          n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        args['mode'] = 'classification'
    elif args['dataset'] == 'External':
        dataset = MoleculeCSVDataset(pd.read_csv(args['external_path']),
                                     smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                                     node_featurizer=args['node_featurizer'],
                                     edge_featurizer=args['edge_featurizer'],
                                     cache_file_path=os.path.join(args['external_path'], 'external_processed'))
        if type(dataset.labels.numpy().squeeze()[0]) == int:
            args['mode'] = 'classification'
        else:
            args['mode'] = 'regression'
    else:
        raise ValueError('Unexpected dataset: {}'.format(args['dataset']))
    args['n_tasks'] = dataset.n_tasks
    return dataset


def cal_diff_feat(args, train_smiles, train_labels):
    label = train_labels
    smiles = train_smiles
    diff_feat = []
    if args['diff_weight']:
        args['diff_weight'] = [float(i) for i in args['diff_weight'].split(' ')]
    if args['diff_type'] in ['LabelDistance', 'Combine_SWL', 'Combine_SWLD']:
        pred = pd.read_csv(args['external_path'])['PREDICT'].values
    else:
        pred = [None] * len(smiles)
    print('Difficult Calculate Method: ', args['diff_type'])
    for idx in range(len(smiles)):
        diff = Feat_Calculate(smiles[idx], args['diff_type'], label[idx], pred[idx])
        if type(diff.diff_feat) == list:
            diff_feat.append(np.sum(np.array(diff.diff_feat) * np.array(args['diff_weight'])))
        else:
            diff_feat.append(diff.diff_feat)
    return np.array(diff_feat)
