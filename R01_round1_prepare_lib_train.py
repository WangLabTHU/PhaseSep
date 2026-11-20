import os
import sys

import pandas as pd

import joblib

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# The two cGAN files, which the project copies to the root beside this one
for _library in (os.path.dirname(os.path.abspath(__file__)), './scripts'):
    if _library and os.path.isdir(_library) and _library not in sys.path:
        sys.path.append(_library)

from _cgan_arch import Deepseed, SAVE_ROOT, open_fa_withidx

# The project layout: the cGAN's training data under ./data/cgan_train, its weights under
# ./checks/cgan_hek293t -- SAVE_ROOT in _cgan_arch.py
DATA_DIR = './data/cgan_train'

HEK293_TRAIN_CSV = os.path.join(DATA_DIR, 'hek293_train.csv')
MASKED_FASTA = os.path.join(DATA_DIR, 'masked_enhancers_HEK293T_240617.txt')
UNMASKED_FASTA = os.path.join(DATA_DIR, 'unmasked_enhancers_HEK293T_240617.txt')

# The model the whole library design is built on
MODEL_NAME = 'deepseed_hek293_debug'
GEN_LENGTH = 252

# The author's training schedule
N_ITERS = 50000
SAVE_ITERS = 1000

# The training table is the file that is copied into ./data; set this True to rebuild it first
# from the two enhancer files, which is what produced it
BUILD_TRAIN_CSV = False

# What the loop below writes last, and what R01_round1_prepare_lib_generate.py loads
GEN_CKPT = os.path.join(SAVE_ROOT, 'net_G_{}.pth'.format(N_ITERS - 1))


def require(path, what):
    '''The file has to be where the path says; say which one is missing.'''
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} is missing. {what}')


if BUILD_TRAIN_CSV:

    masked_seqs, masked_idx = open_fa_withidx(MASKED_FASTA)
    unmasked_seqs, unmasked_idx = open_fa_withidx(UNMASKED_FASTA)
    df = pd.DataFrame({"realA": masked_seqs, "realB": unmasked_seqs})
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(HEK293_TRAIN_CSV)
    print(f'[report] {HEK293_TRAIN_CSV} rebuilt from {MASKED_FASTA} and {UNMASKED_FASTA}, '
          f'{len(df)} sequences')

else:

    require(HEK293_TRAIN_CSV,
            'put the DeepSEED training table there, or set BUILD_TRAIN_CSV = True to build it '
            f'from {MASKED_FASTA} and {UNMASKED_FASTA}')

print(f'[report] training {MODEL_NAME} on {HEK293_TRAIN_CSV}: {N_ITERS} iterations, a checkpoint '
       f'every {SAVE_ITERS}, into {SAVE_ROOT}')


model = Deepseed(n_iters=N_ITERS, save_iters=SAVE_ITERS, dataset=HEK293_TRAIN_CSV, savepath=SAVE_ROOT,
                 length=GEN_LENGTH, model_name=MODEL_NAME)
model.train()

print(f'[report] the generation step reads {GEN_CKPT}')
