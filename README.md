# CurrMG
Codes for "An efficient curriculum learning-based strategy for molecular graph learning"
## Environment Requirement
* torch==1.8.0
* dgl==0.5.2
* dgl-lifesci==0.2.5
* rdkit>=2017.09.1
## Easy Usage
    python main.py -d DATASET -mo MODEL -me METRIC -cu TRUE -rp RESULT_PATH -dt DIFFICULTY_MEASURER
    # optional arguments
    # DATASET: FreeSolv ESOL Lipophilicity BACE BBBP Tox21 ClinTox SIDER External
    # MODEL: GCN GAT MPNN AttentiveFP Pretrained-GIN
    # METRIC: roc_auc_score pr_auc_score r2 mae rmse
    # DIFFICULTY_MEASURER: AtomAndBond Fsp3 MCE18 LabelDistance Joint Two_Stage
    # Find more arguments (e.g. Data splitting, Learning rate, Num epochs, Batch size...), please see main.py
