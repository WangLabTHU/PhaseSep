'''R01: the round-1 predictor -- the three branches trained on the whole library.

tmp/R01_round1_predictor.py with its body, which the author had commented out, run live. The model
is the AttnBiLSTM of _model.py, _model import * giving os, pd, plt, write_fa and write_exp; the
round-1 library is read from the two mean tables, the three branches -- induced, uninduced, lfc --
are trained on all of it, and the checkpoint of each lands under ./checks/round1_design/, which is
where the round-2 design code and S13_mutant_template_boxplot.py look for them. The held-out twin of
this file, which trains the same three branches on nine tenths of the same table, is
S11_train_model.py.

  ./data/7_exp_mean_FUS_TO_induced.tsv      in: the round-1 library, induced arm
  ./data/7_exp_mean_FUS_TO_uninduced.tsv    in: the round-1 library, uninduced arm
  ./data/predictor_train/                   out: train_seqs.txt and the three label files
  ./checks/round1_design/                   out: induced/ uninduced/ lfc/, a checkpoint.pth each

Deviations from tmp/R01_round1_predictor.py:
  1. the body is live. The original held all of it commented out, from the two read_csv calls down
     to the three train_branch calls; what those lines said is what runs, line for line.
  2. paths: the two tables are read from ./data under their renamed project names,
     7_exp_mean_FUS_TO_induced.tsv and 7_exp_mean_FUS_TO_uninduced.tsv, the names the rest of the
     pipeline reads them under. The original read ./datasets/round1_origin/7_exp_mean_induced.tsv
     and ./datasets/round1_origin/7_exp_mean_uninduced.tsv. The two output paths are the author's
     own, ./data/predictor_train/ and ./checks/round1_design/, spelled as constants at the top.
  3. ./data/predictor_train/ is created before the first write. write_fa and write_exp open their
     file themselves and do not make the directory they are given, so with the author's lines alone
     the run stops at the first write unless that directory happens to be there already.
  4. the two input tables are checked before anything is read, so a missing one names itself.
  5. import joblib and the pdf/ps fonttype lines are the standing requirement for delivered files.
     This file draws nothing, so they do nothing here.
  6. open_fa_with_tags, and the re, random, defaultdict and combinations imports, are the author's
     and are not used by this file; they are kept as they were.

Running this file writes ./data/predictor_train/ and trains three branches, 400 epochs each, into
./checks/round1_design/.
'''

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
