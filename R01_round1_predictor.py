from _model import *
import re
import random
from collections import defaultdict
from itertools import combinations

import joblib
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# The project layout: the two mean tables in ./data, what the branches train on and the three
# branches themselves under ./data and ./checks
DATA_DIR = './data'
TRAIN_DIR = './data/predictor_train'

# The library builder leaves a branch's files in the directory it is given, by string
# concatenation, so this one keeps its trailing slash
CHECKPOINT_DIR = './checks/round1_design/'

FILE_INDUCED = '7_exp_mean_FUS_TO_induced.tsv'
FILE_UNINDUCED = '7_exp_mean_FUS_TO_uninduced.tsv'

INDUCED_TSV = os.path.join(DATA_DIR, FILE_INDUCED)
UNINDUCED_TSV = os.path.join(DATA_DIR, FILE_UNINDUCED)

def require(path, what):
    '''The file has to be where the path says; say which one is missing.'''
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} is missing. {what}')


def open_fa_with_tags(file):
    record, tags = [], []
    f = open(file,'r')
    for item in f:
        if '>' not in item:
            record.append(item[0:-1])
        else:
            tags.append(item[1:-1])
    f.close()
    return record, tags

def prepare_datasets(df_merged):

    return (
        df_merged["seqs"].tolist(),
        df_merged["induced_mpra_means"].tolist(),
        df_merged["uninduced_mpra_means"].tolist(),
        df_merged["lfc"].tolist(),
    )

def train_branch(task_name, seqs_file, labels_file, save_dir, length=252, epoch=400, patience=50):
    os.makedirs(save_dir, exist_ok=True)
    model = AttnBilstm_language(length=length, epoch=epoch, patience=patience, model_name=task_name)
    model.train(dataset=seqs_file, labels=labels_file, savepath=save_dir)



# part1: 训练全样本的模型

require(INDUCED_TSV,
        'put the round-1 library mean table there, the induced arm of 7_exp_mean_FUS_TO_*.tsv')
require(UNINDUCED_TSV,
        'put the round-1 library mean table there, the uninduced arm of 7_exp_mean_FUS_TO_*.tsv')

df_induced = pd.read_csv(INDUCED_TSV, sep="\t", header=None)
df_uninduced = pd.read_csv(UNINDUCED_TSV, sep="\t", header=None)
df_induced.columns = ["seqs", "induced_mpra_means", "names"]
df_uninduced.columns = ["seqs", "uninduced_mpra_means", "names"]

df_merged = pd.merge(df_induced, df_uninduced, on=['names', 'seqs'], how='inner')
df_merged = df_merged[["seqs", "names", "induced_mpra_means", "uninduced_mpra_means"]]
df_merged["lfc"] = df_merged["induced_mpra_means"] / df_merged["uninduced_mpra_means"]

print(f'[report] the round-1 library: {len(df_merged)} designs merged from {INDUCED_TSV} and '
      f'{UNINDUCED_TSV}')

train_seqs, train_induced, train_uninduced, train_lfc = prepare_datasets(df_merged)

# write_fa and write_exp do not make the directory they are given, and train_branch makes its own
os.makedirs(TRAIN_DIR, exist_ok=True)

write_fa(os.path.join(TRAIN_DIR, "train_seqs.txt"), train_seqs, df_merged["names"].tolist())
write_exp(os.path.join(TRAIN_DIR, "train_induced.txt"), train_induced)
write_exp(os.path.join(TRAIN_DIR, "train_uninduced.txt"), train_uninduced)
write_exp(os.path.join(TRAIN_DIR, "train_lfc.txt"), train_lfc)

base_dir = TRAIN_DIR
checkpoint_dir = CHECKPOINT_DIR
seqs_file = os.path.join(base_dir, 'train_seqs.txt')

train_branch(
    task_name="induced",
    seqs_file=seqs_file,
    labels_file=os.path.join(base_dir, 'train_induced.txt'),
    save_dir=checkpoint_dir
)

train_branch(
    task_name="uninduced",
    seqs_file=seqs_file,
    labels_file=os.path.join(base_dir, 'train_uninduced.txt'),
    save_dir=checkpoint_dir
)

train_branch(
    task_name="lfc",
    seqs_file=seqs_file,
    labels_file=os.path.join(base_dir, 'train_lfc.txt'),
    save_dir=checkpoint_dir
)

print(f'[report] the three branches -- induced, uninduced, lfc -- are under {checkpoint_dir}')
