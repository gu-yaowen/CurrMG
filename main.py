from argparse import ArgumentParser
from train import train


def main_train():
    parser = ArgumentParser('Molecule Property Prediction')
    # General Arguments
    parser.add_argument('-d', '--dataset', choices=['BACE', 'BBBP', 'ClinTox', 'HIV', 'Tox21',
                                                    'FreeSolv', 'Lipophilicity', 'ESOL', 'SIDER',
                                                    'Clintox', 'Tox21', 'External'],
                        help='Dataset to use')
    parser.add_argument('-ep', '--external_path', default=None, type=str,
                        help='External dataset path (default: None)')
    parser.add_argument('-mo', '--model', choices=['GCN', 'GAT', 'Weave', 'MPNN', 'AttentiveFP',
                                                   'gin_supervised_contextpred'],
                        help='Model to use')
    parser.add_argument('-s', '--split', choices=['scaffold', 'random'], default='scaffold',
                        help='Dataset splitting method (default: scaffold)')
    parser.add_argument('-sr', '--split-ratio', default='0.8,0.1,0.1', type=str,
                        help='Proportion of the dataset to use for training, validation and test, '
                             '(default: 0.8,0.1,0.1)')
    parser.add_argument('-me', '--metric', choices=['roc_auc_score', 'pr_auc_score', 'r2', 'mae', 'rmse'],
                        default='roc_auc_score',
                        help='Metric for evaluation (default: roc_auc_score)')
    parser.add_argument('-pe', '--print-every', type=int, default=50,
                        help='Print the training progress every X mini-batches')
    parser.add_argument('-rp', '--result-path', type=str, default='results',
                        help='Path to save training results (default: results)')
    parser.add_argument('-id', '--cuda_id', type=str, default='0',
                        help='Path to save training results (default: results)')
    parser.add_argument('-se', '--seed', type=int, default=0,
                        help='Global random seed')
    # CurrMG-related Arguments
    parser.add_argument('-cu', '--is_Curr', type=eval, default=True,
                        help='Choose whether to use curriculum learning in training period')
    parser.add_argument('-dt', '--diff_type', type=str, choices=['AtomAndBond', 'Fsp3', 'MCE18',
                                                                 'LabelDistance', 'Joint', 'Two_stage',
                                                                 'None', 'Ablation'],
                        default='AtomAndBond', help='Calculation method of molecular difficulty coefficient to use')
    parser.add_argument('-wt', '--diff_weight', type=float, default=0.6,
                        help='Weight of each difficulty coefficient used(eg: 0.6 when choose Joint)')
    parser.add_argument('-ct', '--c_type', type=int, default=3,
                        help='Power of competence function to use')
    parser.add_argument('-st', '--sample_type', type=str, default='Random',
                        choices=['Random', 'Padding-like'],
                        help='Way of sample type to generate a mini batch data')
    # General Training Arguments
    parser.add_argument('-ne', '--num_epochs', type=int, default=100,
                        help='Number of epochs for training')
    parser.add_argument('-lr', '--learning_rate', type=float,
                        help='learning rate to use')
    parser.add_argument('-bs', '--batch_size', type=int,
                        help='batch size to use')
    parser.add_argument('-wd', '--weight_decay', type=float,
                        help='weight decay to use')
    # Optional Arguments
    parser.add_argument('-th', '--threshold', type=float, default=1.0,
                        help='The threshold of competence function (Only for ablation study)')
    parser.add_argument('-rt', '--ratio', type=float, default=1.0,
                        help='The ratio of training data to use (Only for ablation study)')
    parser.add_argument('-nw', '--num-workers', type=int, default=0,
                        help='Number of processes for data loading (default: 0)')
    parser.add_argument('-f', '--featurizer-type', choices=['canonical', 'attentivefp'], default='canonical',
                        help='Featurization for atoms (and bonds). This is required for models '
                             'other than gin_supervised_**.')
    parser.add_argument('-p', '--pretrain', action='store_true',
                        help='Whether to skip the training and evaluate the pre-trained model '
                             'on the test set (default: False)')
    args = parser.parse_args().__dict__
    # print(args)
    train(args)


if __name__ == '__main__':
    main_train()
