# CurrMG
Codes for "An efficient curriculum learning-based strategy for molecular graph learning"
# Reference
If you make advantage of the CurrMG training strategy proposed in our paper, please cite the following in your manuscript:

```
@article{10.1093/bib/bbac099,
author = {Gu, Yaowen and Zheng, Si and Xu, Zidu and Yin, Qijin and Li, Liang and Li, Jiao},
title = "{An efficient curriculum learning-based strategy for molecular graph learning}",
journal = {Briefings in Bioinformatics},
year = {2022},
month = {04},
issn = {1477-4054},
doi = {10.1093/bib/bbac099},
}
```
# Overview
![CurrMG](https://github.com/gu-yaowen/CurrMG/blob/main/overview.png)
## Environment Requirement
* torch==1.8.0
* dgl==0.5.2
* dgl-lifesci==0.2.5
* rdkit>=2017.09.1
## Easy Usage
    python main.py -d {DATASET} -mo {MODEL} -me {METRIC} -cu TRUE -rp {SAVED PATH} -dt {DIFFICULTY MEASURER}
    Main arguments:
        -d: FreeSolv ESOL Lipophilicity BACE BBBP Tox21 ClinTox SIDER External(your own dataset)
        -mo: GCN GAT MPNN AttentiveFP Pretrained-GIN
        -me: roc_auc_score pr_auc_score r2 mae rmse
        -dt: AtomAndBond Fsp3 MCE18 LabelDistance Joint Two_stage
    Optional arguments:
        -s: Random or scaffold splitting type.
        -sr: Split ratio.
        -wt: Weight of difficulty coefficient for d_Joint and d_Two_stage.
        -ct: Power of competence function.
        -ne: Epoches. -lr: Learning rate. -bs: Batch Size. -wd: Weight decay.
    For more arguments, please see main.py
## Sample data
Once you want to use your own dataset, please follow the file format as ``JAK2.csv`` and ``Mtb.csv`` in 'test' folder.
# Contact
We welcome you to contact us (email: gu.yaowen@imicams.ac.cn) for any questions and cooperations.
