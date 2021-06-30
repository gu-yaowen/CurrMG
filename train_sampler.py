import numpy as np
from torch.utils.data import Sampler
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors


def read_smiles(smi):
    """
    Read SMILES from the input file.
    Args:
        smi: str,
        The SMILES of the string

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
        Input rdkit mol object and the feature calculation option
        Args:
            extra: float or None
                The extra feature to be calculated for the difficulty.
            mol: rdkit.Mol object
                The mol whose feature to be calculated.
            curr_option: str, The option for the Curr_learning, choice
                from [AtomAndBond, Fsp3Ring, MCE18, LabelDistance, Combine]
                The string of the choice for the feature calculation
        """
        self.mol = read_smiles(smiles)
        self.curr_option = curr_option
        if self.curr_option == 'AtomAndBond':
            self.diff_feat = self.calculate_atom_and_bond()
        elif self.curr_option == 'Fsp3Ring':
            self.diff_feat = self.calculate_fsp3ring()
        elif self.curr_option == 'MCE18':
            self.diff_feat = self.calculate_MCE18()
        elif self.curr_option == 'LabelDistance':
            self.label = label
            self.pred = pred
            self.diff_feat = self.calculate_LabelDistance()
        elif self.curr_option == 'Combine_SWL':
            self.label = label
            self.pred = pred
            self.diff_feat = [self.calculate_atom_and_bond(),
                              self.calculate_MCE18(), self.calculate_LabelDistance()]
        elif self.curr_option == 'Combine_S':
            self.diff_feat = [self.calculate_atom_and_bond(),
                              self.calculate_MCE18()]
        elif self.curr_option == 'Combine_SWLD':
            self.label = label
            self.pred = pred
            self.diff_feat = [self.calculate_atom_and_bond(),
                              self.calculate_MCE18(),
                              self.calculate_LabelDistance()]
        elif self.curr_option == 'None':
            self.diff_feat = []
        else:
            self.diff_feat = None

    def calculate_atom_and_bond(self):
        """
        Calculate the summation of the atom number and bond number
        Returns:

        """
        return self.mol.GetNumAtoms() + self.mol.GetNumBonds()

    def calculate_chiral(self):
        """
        Calculate the number of the chiral center of the molecule.
        Returns:

        """
        Chem.AssignStereochemistry(self.mol, flagPossibleStereoCenters=True)
        return rdMolDescriptors.CalcNumAtomStereoCenters(self.mol)

    def calculate_fsp3ring(self):
        """
        Calculate the Fsp3 ration in all the rings in the molecule.
        Returns:

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

        """
        return abs(self.pred - self.label)


def diff_metric_get(diff_count):
    """
    To get the difficulty metric of the difficulty metric
    Args:
        diff_count: The array of the difficulty score of the

    Returns:

    """
    if type(diff_count[0]) == list:
        diff_count = np.stack(diff_count)
        cdf = np.zeros(len(diff_count))
        weight = [0.3, 0.2, 0.5]
        count = 0
        for diff in diff_count.T:
            cdf_ = np.array([len(np.where(diff < count)[0]) / len(diff) for count in diff])
            cdf += cdf_ * weight[count]
            count += 1
        sort = cdf.argsort()
    else:
        sort = diff_count.argsort()
        # diff_count = diff_count[diff_count.argsort()]
        cdf = np.array([len(np.where(diff_count < count)[0]) /
                        len(diff_count) for count in diff_count])
    return sort, cdf


def competence_func(t_step: int, t_total: int, c0: float, c_type: float):
    competence = pow((1 - c0 ** c_type) * (t_step / t_total) + c0 ** c_type, 1 / c_type)
    if competence > 1:
        competence = 1
    return competence


class CurrSampler(Sampler):
    """
    The sampler based on the CurrLearning.
    """

    def __init__(self, diff_feat):
        self.diff_feat = diff_feat

    def __iter__(self):
        self.indices, self.cdf_dis = diff_metric_get(self.diff_feat)
        return iter([[self.indices, self.cdf_dis]])

    def __len__(self):
        return len(self.indices)


class CurrBatchSampler(Sampler):
    r"""The batch sampler for the data sample"""

    def __init__(self, sampler, batch_size, t_total, c_type, random_state, sample_type):
        """

        Args:
            sampler: torch.utils.data.Sampler,
            The defined Sampler for the data sample, return the
            batch_size:
            t_total:
            c0:
            c_type:
            random_state:
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.t_total = t_total
        self.c_type = c_type
        self.random_state = random_state
        self.sample_type = sample_type

    def __iter__(self):
        for sample in self.sampler:
            self.indices = np.array(sample[0])
            self.cdf = np.array(sample[1])
        self.c0 = np.argpartition(self.cdf, self.batch_size - 1)[self.batch_size]
        sample_count = np.zeros(len(self.indices))
        np.random.seed(self.random_state)
        for t in range(self.t_total):
            c = competence_func(t, self.t_total, self.c0, self.c_type)
            sample_pool = list(np.where(self.cdf <= c)[0])
            if self.sample_type == 'Random':
                sample_all = np.random.choice(sample_pool, size=self.batch_size, replace=False)
            elif self.sample_type == 'Padding-like':
                sample_all = np.argpartition(sample_count[sample_pool],
                                             self.batch_size - 1)[:self.batch_size]
                sample_count[sample_all] += 1
            # print(len(sample_pool))
            yield sample_all.tolist()

    def __len__(self):
        return len(self.indices)
