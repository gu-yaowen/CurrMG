import json
import os


def set_model_config(args):
    if args['model'] == 'gin_supervised_contextpred':
        config = {
            "batch_size": 128,
            "jk": "concat",
            "lr": 0.001,
            "patience": 30,
            "readout": "sum",
            "weight_decay": 0.001
        }
    elif args['model'] == 'GCN':
        config = {
            "batch_size": 128,
            "batchnorm": False,
            "dropout": 0.1,
            "gnn_hidden_feats": 128,
            "lr": 0.002,
            "num_gnn_layers": 2,
            "patience": 30,
            "predictor_hidden_feats": 64,
            "residual": True,
            "weight_decay": 0.001
        }
    elif args['model'] == 'GAT':
        config = {
            "alpha": 0.5,
            "batch_size": 128,
            "dropout": 0.05,
            "gnn_hidden_feats": 128,
            "lr": 0.01,
            "num_gnn_layers": 2,
            "num_heads": 6,
            "patience": 30,
            "predictor_hidden_feats": 128,
            "residual": False,
            "weight_decay": 0.0005
        }
    elif args['model'] == 'MPNN':
        config = {
            "batch_size": 128,
            "edge_hidden_feats": 64,
            "lr": 0.001,
            "node_out_feats": 48,
            "num_layer_set2set": 2,
            "num_step_message_passing": 2,
            "num_step_set2set": 2,
            "patience": 30,
            "weight_decay": 0.0005
        }
    elif args['model'] == 'AttentiveFP':
        config = {
            "batch_size": 128,
            "dropout": 0.2,
            "graph_feat_size": 32,
            "lr": 0.01,
            "num_layers": 2,
            "num_timesteps": 3,
            "patience": 30,
            "weight_decay": 0.001
        }
    return config


def GCN_config(args, config):
    if args['dataset'] == 'FreeSolv':
        config['lr'] = 0.001
        config['batch_size'] = 64
        config['weight_decay'] = 0.0
    elif args['dataset'] == 'ESOL':
        config['lr'] = 0.001
        config['batch_size'] = 64
        config['weight_decay'] = 0.0
    elif args['dataset'] == 'Lipophilicity':
        config['lr'] = 0.001
        config['batch_size'] = 64
        config['weight_decay'] = 0.0
    elif args['dataset'] == 'BBBP':
        config['lr'] = 0.001
        config['batch_size'] = 256
        config['weight_decay'] = 0.0005
    elif args['dataset'] == 'BACE':
        config['lr'] = 0.001
        config['batch_size'] = 256
        config['weight_decay'] = 0.0005
    elif args['dataset'] == 'HIV':
        config['lr'] = 0.004
        config['batch_size'] = config['batch_size']
        config['weight_decay'] = config['weight_decay']
    return config


def GAT_config(args, config):
    if args['dataset'] == 'FreeSolv':
        config['lr'] = 0.001
        config['batch_size'] = 256
        config['weight_decay'] = 0.0005
    elif args['dataset'] == 'ESOL':
        config['lr'] = 0.001
        config['batch_size'] = 256
        config['weight_decay'] = 0.0005
    elif args['dataset'] == 'Lipophilicity':
        config['lr'] = 0.0005
        config['batch_size'] = 64
        config['weight_decay'] = 0.0
    elif args['dataset'] == 'BBBP':
        config['lr'] = 0.0005
        config['batch_size'] = 64
        config['weight_decay'] = 0.0
    elif args['dataset'] == 'BACE':
        config['lr'] = 0.001
        config['batch_size'] = 64
        config['weight_decay'] = 0.0
    elif args['dataset'] == 'HIV':
        config['lr'] = 0.004
        config['batch_size'] = config['batch_size']
        config['weight_decay'] = config['weight_decay']
    return config


def MPNN_config(args, config):
    if args['dataset'] == 'FreeSolv':
        config['lr'] = 0.005
        config['batch_size'] = 128
        config['weight_decay'] = 0.0005
    elif args['dataset'] == 'ESOL':
        config['lr'] = 0.001
        config['batch_size'] = 64
        config['weight_decay'] = 0.0
    elif args['dataset'] == 'Lipophilicity':
        config['lr'] = 0.001
        config['batch_size'] = 64
        config['weight_decay'] = 0.0005
    elif args['dataset'] == 'BBBP':
        config['lr'] = 0.005
        config['batch_size'] = 64
        config['weight_decay'] = 0.0005
    elif args['dataset'] == 'BACE':
        config['lr'] = 0.0005
        config['batch_size'] = 64
        config['weight_decay'] = 0.0005
    elif args['dataset'] == 'HIV':
        config['lr'] = config['batch_size']
        config['batch_size'] = config['batch_size']
        config['weight_decay'] = config['weight_decay']
    return config


def AttentiveFP_config(args, config):
    if args['dataset'] == 'FreeSolv':
        config['lr'] = 0.01
        config['batch_size'] = 64
        config['weight_decay'] = 0.0005
    elif args['dataset'] == 'ESOL':
        config['lr'] = 0.005
        config['batch_size'] = 64
        config['weight_decay'] = 0.0
    elif args['dataset'] == 'Lipophilicity':
        config['lr'] = 0.005
        config['batch_size'] = 64
        config['weight_decay'] = 0.0
    elif args['dataset'] == 'BBBP':
        config['lr'] = 0.005
        config['batch_size'] = 64
        config['weight_decay'] = 0.0
    elif args['dataset'] == 'BACE':
        config['lr'] = 0.005
        config['batch_size'] = 64
        config['weight_decay'] = 0.0005
    elif args['dataset'] == 'HIV':
        config['lr'] = config['batch_size']
        config['batch_size'] = config['batch_size']
        config['weight_decay'] = config['weight_decay']
    return config


def Pretrained_GIN_config(args, config):
    if args['dataset'] == 'FreeSolv':
        config['lr'] = 0.001
        config['batch_size'] = 256
        config['weight_decay'] = 0.001
    elif args['dataset'] == 'ESOL':
        config['lr'] = 0.0005
        config['batch_size'] = 256
        config['weight_decay'] = 0.001
    elif args['dataset'] == 'Lipophilicity':
        config['lr'] = 0.0005
        config['batch_size'] = 128
        config['weight_decay'] = 0.0
    elif args['dataset'] == 'BBBP':
        config['lr'] = 0.0005
        config['batch_size'] = 64
        config['weight_decay'] = 0.0005
    elif args['dataset'] == 'BACE':
        config['lr'] = 0.001
        config['batch_size'] = 128
        config['weight_decay'] = 0.0
    elif args['dataset'] == 'HIV':
        config['lr'] = config['batch_size']
        config['batch_size'] = config['batch_size']
        config['weight_decay'] = config['weight_decay']
    return config
