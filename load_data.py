import os
import numpy as np
import pandas as pd
from dgllife.data import FreeSolv, Lipophilicity, ESOL, \
    BBBP, BACE, ClinTox, HIV, Tox21, SIDER
from functools import partial
from dgllife.utils import smiles_to_bigraph
from dgllife.data.csv_dataset import MoleculeCSVDataset
from torch.utils.data import DataLoader
from utils import collate_molgraphs
from train_sampler import CurrSampler, CurrBatchSampler


def load_data_from_dgl(args):
    """
    Load molecular property prediction datasets. In there, some pre-prepared datasets
    have been sotred in DGL-life. Note that once you want to use your own External Dataset,
    please follow the file format as those in "test" folder.
    Returns:
        A dgl-type dataset.
    """
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
    elif args['dataset'] == 'Clintox':
        dataset = ClinTox(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                          node_featurizer=args['node_featurizer'],
                          edge_featurizer=args['edge_featurizer'],
                          n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        args['mode'] = 'classification'
    elif args['dataset'] == 'SIDER':
        dataset = SIDER(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                        node_featurizer=args['node_featurizer'],
                        edge_featurizer=args['edge_featurizer'],
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        args['mode'] = 'classification'
    elif args['dataset'] == 'External':
        dataset = MoleculeCSVDataset(pd.read_csv(args['external_path']),
                                     smiles_column='SMILES',
                                     smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                                     node_featurizer=args['node_featurizer'],
                                     edge_featurizer=args['edge_featurizer'],
                                     cache_file_path=os.path.join(args['result_path'], 'external_processed'))
        if dataset.labels.numpy().squeeze()[0] % 1 == 0.0:
            if len(np.where(dataset.labels.numpy().squeeze() > 1)[0]) > 1:
                args['n_tasks'] = len(np.unique(dataset.labels.numpy().squeeze()))
                args['mode'] = 'multi-label'
            else:
                args['mode'] = 'classification'
                args['n_tasks'] = 1
        else:
            args['mode'] = 'regression'
    else:
        raise ValueError('Unexpected dataset: {}'.format(args['dataset']))
    if args['dataset'] != 'External':
        args['n_tasks'] = dataset.n_tasks
    return dataset


def load_data(args, train_set, val_set, test_set,
              diff_feat: np.array):
    """
    Build data loader for model training.
    Args:
        train_set: The training set.
        val_set: The validating set.
        test_set: The test set.
        diff_feat: Difficulty coefficients for training set.
    Returns:
        The training set data loader, validating data loader, and test data loader.
    """
    if args['is_Curr']:
        print('Training Method in Curriculum Learning')
        sampler = CurrSampler(args, diff_feat)
        batch_sampler = CurrBatchSampler(sampler, args['batch_size'],
                                         args['t_total'], args['c_type'],
                                         args['sample_type'], args['threshold'])
        train_loader = DataLoader(train_set,
                                  batch_sampler=batch_sampler,
                                  num_workers=args['num_workers'],
                                  collate_fn=collate_molgraphs)
    else:
        print('Training Method NOT in Curriculum Learning')
        train_loader = DataLoader(train_set, batch_size=args['batch_size'],
                                  num_workers=args['num_workers'],
                                  shuffle=True, collate_fn=collate_molgraphs)
    if val_set:
        val_loader = DataLoader(val_set,
                                batch_size=int(len(val_set) * 0.2) if int(len(val_set) * 0.2) < 1000 else 1000,
                                num_workers=args['num_workers'], collate_fn=collate_molgraphs)
    else:
        val_loader = None
    test_loader = DataLoader(test_set,
                             batch_size=int(len(test_set) * 0.2) if int(len(test_set) * 0.2) < 1000 else 1000,
                             num_workers=args['num_workers'], collate_fn=collate_molgraphs)
    return train_loader, val_loader, test_loader
