from argparse import ArgumentParser
from train import train


def main_train():
    parser = ArgumentParser('Molecule Property Prediction')
    parser.add_argument('-d', '--dataset', choices=['BACE', 'BBBP', 'ClinTox', 'HIV', 'Tox21',
                                                    'FreeSolv', 'Lipophilicity', 'ESOL', 'External'],
                        help='Dataset to use')
    parser.add_argument('-ep', '--external_path', default=None, type=str,
                        help='External dataset path (default: None)')
    parser.add_argument('-mo', '--model', choices=['GCN', 'GAT', 'MPNN', 'AttentiveFP',
                                                   'gin_supervised_contextpred'],
                        help='Model to use')
    parser.add_argument('-f', '--featurizer-type', choices=['canonical', 'attentivefp'], default='canonical',
                        help='Featurization for atoms (and bonds). This is required for models '
                             'other than gin_supervised_**.')
    parser.add_argument('-p', '--pretrain', action='store_true',
                        help='Whether to skip the training and evaluate the pre-trained model '
                             'on the test set (default: False)')
    parser.add_argument('-s', '--split', choices=['scaffold', 'random'], default='scaffold',
                        help='Dataset splitting method (default: scaffold)')
    parser.add_argument('-sr', '--split-ratio', default='0.8,0.1,0.1', type=str,
                        help='Proportion of the dataset to use for training, validation and test, '
                             '(default: 0.8,0.1,0.1)')
    parser.add_argument('-me', '--metric', choices=['roc_auc_score', 'pr_auc_score', 'r2', 'mae', 'rmse'],
                        default='roc_auc_score',
                        help='Metric for evaluation (default: roc_auc_score)')
    parser.add_argument('-n', '--num-epochs', type=int, default=100,
                        help='Maximum number of epochs for training. '
                             'We set a large number by default as early stopping '
                             'will be performed. (default: 100)')
    parser.add_argument('-nw', '--num-workers', type=int, default=0,
                        help='Number of processes for data loading (default: 0)')
    parser.add_argument('-pe', '--print-every', type=int, default=50,
                        help='Print the training progress every X mini-batches')
    parser.add_argument('-rp', '--result-path', type=str, default='results',
                        help='Path to save training results (default: results)')
    parser.add_argument('-id', '--cuda_id', type=str, default='0',
                        help='Path to save training results (default: results)')
    parser.add_argument('-cu', '--is_Curr', type=eval, default=True,
                        help='Choose whether to use curriculum learning in training period')
    parser.add_argument('-dt', '--diff_type', type=str, choices=['AtomAndBond', 'MCE18', 'LabelDistance',
                                                                 'Combine_S', 'Combine_SWL', 'Combine_SWLD'],
                        default='AtomAndBond', help='Calculation method of molecular difficulty coefficient to use')
    parser.add_argument('-wt', '--diff_weight', type=str, default=None,
                        help='Weight of each difficulty coefficient used(eg: "0.2 0.8" when choose Combine_S)')
    parser.add_argument('-ct', '--c_type', type=int, default=4,
                        help='Power of competence function to use')
    parser.add_argument('-st', '--sample_type', type=str, default='Random',
                        choices=['Random', 'Padding-like'],
                        help='Way of sample type to generate a mini batch data')
    parser.add_argument('-se', '--seed', type=int, default=0,
                        help='Global random seed')

    args = parser.parse_args().__dict__
    print(args)
    train(args)


if __name__ == '__main__':
    main_train()
