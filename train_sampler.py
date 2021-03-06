import random
import numpy as np
from torch.utils.data import Sampler
from rdkit import Chem
import copy
from torch.optim import Adam
from rdkit.Chem import AllChem, rdMolDescriptors
from utils import load_model, predict, criterion
from dgllife.utils import RandomSplitter


def cal_diff_feat(args, dataset, train_set):
    """
    Calculate difficulty coefficients for training set based on selected difficulty measurer.
    Args:
        dataset: The whole dataset.
        train_set: The training set.
    Returns:
        The numpy array of difficulty coefficients for training set.
    """
    smiles = np.array(dataset.smiles)[train_set.indices]
    label = dataset.labels.numpy().squeeze()[train_set.indices]
    diff_feat = []
    if args['diff_type'] in ['LabelDistance', 'Joint', 'Two_stage']:
        pred = train4LabelDistance(args, train_set)
    else:
        pred = [None] * len(smiles)
    print('Difficult Calculate Method: ', args['diff_type'])
    for idx in range(len(smiles)):
        diff = Feat_Calculate(smiles[idx], args['diff_type'], label[idx], pred[idx])
        diff_feat.append(diff.diff_feat)
    return np.array(diff_feat)


def train4LabelDistance(args, train_set):
    """
    Acquire the predictions of training set which is used in d_LabelDistance difficulty measurer.
    Args:
        train_set: The training set.
    Returns:
        The prediction results of k teacher models.
    """
    from load_data import load_data
    from train import eval_iteration
    print('LabelDistance Training...')
    args_ = copy.deepcopy(args)
    args_['is_Curr'] = False
    model = load_model(args).to(args_['device'])
    loss_criterion = criterion(args_)
    optimizer = Adam(model.parameters(), lr=args_['lr'],
                     weight_decay=args_['weight_decay'])
    if args['n_tasks'] > 1:
        pred = np.zeros((len(train_set), args['n_tasks']))
    else:
        pred = np.zeros(len(train_set))
    for train, test in RandomSplitter.k_fold_split(train_set, k=3,
                                                   random_state=args_['seed']):
        print('length of train data:', len(train))
        args_['t_total'] = int(100 * len(train) / args['batch_size'])
        train_loader, _, test_loader = load_data(args_, train,
                                                 None, test, None)
        model.train()
        iter_conut = 0
        for i in range(999):
            if iter_conut == args['t_total']:
                break
            for batch_id, batch_data in enumerate(train_loader):
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
                model.train()
                if iter_conut % int(len(train) / 5) == 0:
                    print('iteration {:d}/{:d}, loss {:.4f}'.format(
                        iter_conut, args_['t_total'], loss.item()))
                if iter_conut == args_['t_total']:
                    break
                iter_conut += 1
        _, pred[test.indices] = eval_iteration(args, model, test_loader)
    return pred


def read_smiles(smi):
    """
    Read SMILES from the input file.
    Args:
        smi: The SMILES of the string
    Returns:
        The rdkit rdMol object based on the input SMILES.
    """
    rdkit_mol = AllChem.MolFromSmiles(smi)
    if rdkit_mol is None:
        rdkit_mol = AllChem.MolFromSmiles(smi, sanitize=False)
    return rdkit_mol


class Feat_Calculate:
    def __init__(self, smiles, curr_option, label, pred):
        """
        Input SMILES strings and the difficulty coefficient calculation option
        Args:
            smiles: The SMILES whose difficulty coefficient to be calculated.
            curr_option: The option for the Curr_learning, choice
                from [AtomAndBond, Fsp3, MCE18, LabelDistance, Joint, Two_Stage]
                The string of the choice for the feature calculation
            label: The true label of training set for d_LabelDistance.
            pred: The predictions of training set for d_LabelDistance.
        """
        self.mol = read_smiles(smiles)
        self.label = label
        self.pred = pred
        self.curr_option = curr_option
        if self.curr_option == 'AtomAndBond':
            self.diff_feat = self.calculate_atom_and_bond()
        elif self.curr_option == 'Fsp3':
            self.diff_feat = self.calculate_sp3idx()
        elif self.curr_option == 'MCE18':
            self.diff_feat = self.calculate_MCE18()
        elif self.curr_option == 'LabelDistance':
            self.label = label
            self.pred = pred
            self.diff_feat = self.calculate_LabelDistance()
        elif self.curr_option in ['Joint', 'Two_stage']:
            self.diff_feat = [self.calculate_atom_and_bond(),
                              self.calculate_sp3idx(),
                              self.calculate_MCE18(),
                              self.calculate_LabelDistance()]
        elif self.curr_option == 'None':
            self.diff_feat = []
        elif self.curr_option == 'Ablation':
            self.diff_feat = random.random()
        else:
            self.diff_feat = None

    def calculate_atom_and_bond(self):
        """
        Calculate the summation of the atom number and bond number
        Returns:
            The difficulty coefficient calculated by d_AtomAndBond.
        """
        return self.mol.GetNumAtoms() + self.mol.GetNumBonds()

    def calculate_sp3idx(self):
        """
        Calculate the content of sp3 carbon atoms in the molecule.
        Returns:
            The difficulty coefficient calculated by d_Fsp3.
        """
        n_carbon = 0
        n_sp3ring = 0
        for atom in self.mol.GetAtoms():
            if atom.GetAtomicNum() == 6:
                n_carbon += 1
                if atom.GetTotalDegree() == 4:
                    n_sp3ring += 1
        if not n_carbon:
            return 0
        else:
            return n_sp3ring / n_carbon

    def calculate_chiral(self):
        """
        Calculate the number of the chiral center of the molecule
        for the calculation of d_MCE18.
        """
        Chem.AssignStereochemistry(self.mol, flagPossibleStereoCenters=True)
        return rdMolDescriptors.CalcNumAtomStereoCenters(self.mol)

    def calculate_fsp3ring(self):
        """
        Calculate the Fsp3 ration in all the rings in the molecule
        for the calculation of d_MCE18.
        """
        ring_atoms = [i for ring in self.mol.GetRingInfo().AtomRings() for i in ring]
        n_carbon = 0
        n_sp3ring = 0
        for atom_id in ring_atoms:
            atom = self.mol.GetAtomWithIdx(atom_id)
            if atom.GetAtomicNum() == 6:
                n_carbon += 1
                if atom.GetTotalDegree() == 4:
                    n_sp3ring += 1
        if not n_carbon:
            return 0
        else:
            return n_sp3ring / n_carbon

    def calculate_MCE18(self):
        """
        Calculate the MCE18 score, which is the measure of the complexity.
        Returns:
            The difficulty coefficient calculated by d_MCE18.
        """
        QINDEX = 3 + sum((atom.GetDegree() ** 2) / 2 - 2 for atom in self.mol.GetAtoms())
        FSP3 = rdMolDescriptors.CalcFractionCSP3(self.mol)
        AR = AllChem.CalcNumAromaticRings(self.mol)
        SPIRO = rdMolDescriptors.CalcNumSpiroAtoms(self.mol)
        NRING = rdMolDescriptors.CalcNumRings(self.mol)
        FSP3RING = self.calculate_fsp3ring()
        CHIRALC = self.calculate_chiral()
        return QINDEX * (2 * FSP3RING / (1 + FSP3) + int(AR > 0) +
                         int(AR < NRING) + int(CHIRALC > 0) + int(SPIRO > 0))

    def calculate_LabelDistance(self):
        """
        Calculate the L1 distance of predict value and true label in train set,
        which is the measure of the complexity.
        Returns:
            The difficulty coefficient calculated by d_LabelDistance.
        """
        if type(self.pred) == np.ndarray or type(self.pred) != np.float64:
            return (np.array(self.pred) - np.array(self.label)).mean(axis=0)
        return abs(self.pred - self.label)


def diff_metric_get(args, diff_count):
    """
    To get the normalized difficulty coefficients and the sort of training set.
    Args:
        diff_count: The array of the difficulty coefficients of the training set.
    Returns:
        sort: A sort of training set based on its difficulty coefficients.
        cdf: normalized difficulty coefficients.
    """
    if args['diff_type'] == 'Joint':
        diff_count = np.stack(diff_count)
        cdf = []
        weight = args['diff_weight']
        count = 0
        for i in range(len(diff_count[0])):
            cdf.append(np.array([len(np.where(
                diff_count[:, i] < count)[0]) / len(diff_count[:, i])
                                 for count in diff_count[:, i]]))
            # for ablation study to use
            # cdf.append(np.array([(count - diff_count.min())
            #            / (diff_count.max() - diff_count.min())
            #                      for count in diff_count[:, i]]))
            count += 1
        cdf = np.array(cdf).T
        if args['diff_type'] == 'Joint':
            cdf = np.array([weight * (0.3 * i[0] + 0.2 * i[1] + 0.5 * i[2]) +
                            (1 - weight) * i[3]
                            for i in cdf])
            cdf = np.array([(i - cdf.min()) /
                            (cdf.max() - cdf.min()) for i in cdf])
        else:
            cdf = np.array([0.3 * i[0] + 0.2 * i[1] + 0.5 * i[2]
                            for i in cdf])
            cdf = np.array([(i - cdf.min()) /
                            (cdf.max() - cdf.min()) for i in cdf])
        sort = cdf.argsort()
        return sort, cdf
    elif args['diff_type'] == 'Two_stage':
        diff_count = np.stack(diff_count)
        cdf = []
        weight = args['diff_weight']
        count = 0
        for i in range(len(diff_count[0])):
            cdf.append(np.array([len(np.where(
                diff_count[:, i] < count)[0]) / len(diff_count[:, i])
                                 for count in diff_count[:, i]]))
            # for ablation study to use
            # cdf.append(np.array([(count - diff_count.min())
            #            / (diff_count.max() - diff_count.min())
            #                      for count in diff_count[:, i]]))

            count += 1
        cdf = np.array(cdf).T
        cdf1 = np.array([0.0 * i[0] + 0.0 * i[1] + 1.0 * i[2]
                         for i in cdf])
        cdf2 = cdf[:, -1]
        sort1 = cdf1.argsort()
        sort2 = cdf2.argsort()
        return [sort1, sort2], [cdf1, cdf2]
    else:
        sort = diff_count.argsort()
        cdf = np.array([len(np.where(diff_count < count)[0]) /
                        len(diff_count) for count in diff_count])
        # for ablation study to use
        # cdf = np.array([(count - diff_count.min())
        #                 / (diff_count.max() - diff_count.min())
        #                 for count in diff_count])
        return sort, cdf


def competence_func(t_step: int, t_total: int, c0: float,
                    c_type: float, threshold=1.0):
    """
    The competence-based training scheduler.
        Args:
            t_step: The current training iteration t.
            t_total: Total training iterations T.
            c0: Initial competence value.
            c_type: The power of the number in competence function.
            threshold: only for ablation study.
        Returns:
            Current competence value.
    """
    competence = pow((1 - c0 ** c_type) * (t_step / t_total) + c0 ** c_type, 1 / c_type)
    if competence > threshold:
        competence = threshold
    return competence


class CurrSampler(Sampler):
    """
    The sampler based on the CurrMG.
    """

    def __init__(self, args, diff_feat):
        self.args = args
        self.diff_feat = diff_feat

    def __iter__(self):
        self.indices, self.cdf_dis = diff_metric_get(self.args, self.diff_feat)
        return iter([[self.indices, self.cdf_dis]])

    def __len__(self):
        return len(self.indices)


class CurrBatchSampler(Sampler):
    """
    The batch sampler for the data sample in CurrMG.
    """

    def __init__(self, sampler, batch_size, t_total,
                 c_type, sample_type, threshold=1.0):
        """
        Args:
            sampler: torch.utils.data.Sampler,
            The defined Sampler for the data sample, return the
            batch_size: Batch size.
            t_total: Total training iterations T.
            c_type: The power of the number in competence function.
            sample_type: 'Random' or 'Padding-like' sampling type
            ('Random' is used in our manuscript).
            threshold: only for ablation study.
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.t_total = t_total
        self.c_type = c_type
        self.sample_type = sample_type
        self.threshold = threshold

    def __iter__(self):
        for sample in self.sampler:
            self.indices = np.array(sample[0])
            self.cdf = np.array(sample[1])

        # for ablation study to use
        # self.c0 = 1

        for t in range(self.t_total):
            sample_count = np.zeros(len(self.indices))
            if len(self.indices) > 2:
                self.c0 = self.cdf[np.argpartition(self.cdf, self.batch_size - 1)[self.batch_size]]
                c = competence_func(t, self.t_total, self.c0, self.c_type, self.threshold)
                sample_pool = list(np.where(self.cdf <= c)[0])
                if self.sample_type == 'Random':
                    sample_all = Random_batch(sample_pool,
                                              self.batch_size)
                elif self.sample_type == 'Padding-like':
                    sample_all, sample_count = PaddingLike_batch(sample_pool,
                                                                 sample_count,
                                                                 self.batch_size)
            elif len(self.indices) == 2:
                if t < int(self.t_total * 0.6):
                    self.c0 = self.cdf[0][np.argpartition(self.cdf[0], self.batch_size - 1)[self.batch_size]]
                    c = competence_func(t, int(self.t_total * 0.6),
                                        self.c0, self.c_type, self.threshold)
                    sample_pool = list(np.where(self.cdf[0] <= c)[0])
                    sample_all = Random_batch(sample_pool,
                                              self.batch_size)
                else:
                    self.c0 = self.cdf[1][np.argpartition(self.cdf[1], self.batch_size - 1)[self.batch_size]]
                    c = competence_func(t, int(self.t_total * 0.4),
                                        self.c0, self.c_type, self.threshold)
                    sample_pool = list(np.where(self.cdf[1] <= c)[0])
                    sample_all = Random_batch(sample_pool,
                                              self.batch_size)
            yield sample_all.tolist()

    def __len__(self):
        return len(self.indices)


def Random_batch(sample_pool, batch_size):
    """
    The 'Random' sampling type.
    """
    return np.random.choice(sample_pool, size=batch_size, replace=False)


def PaddingLike_batch(sample_pool, sample_count, batch_size):
    """
    The 'Padding-like' sampling type.
    """
    sample_all = np.argpartition(sample_count[sample_pool],
                                 batch_size - 1)[:batch_size]
    sample_count[sample_all] += 1
    return sample_all, sample_count
