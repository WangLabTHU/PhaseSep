'''Step 1 of the round-1 library design: train the DeepSEED enhancer model on the HEK293T enhancers.

The first of the three steps that R01_round1_prepare_lib.py used to hold as one file. The model is
a WGAN-GP generator over 252 bp, trained to fill the masked positions of a HEK293T enhancer with
what the unmasked ones look like; the library generation of step 2 then asks it to fill the same
mask over the tetO templates. This is the block the author's cgan_debug.py had commented out under
"## done: training".

  ./checks/cgan_hek293t/net_G_49999.pth               the weights, pretrained or trained here
  ./checks/cgan_hek293t/cache/deepseed_hek293_debug/  log, figures, per-iteration samples

The pretrained net_G_49999.pth the author's own run left behind goes to the same path, flat in the
weights directory; that file and the one this file trains are one path, so running this file with
the pretrained weights already there would overwrite them.

Deviations from cgan_debug.py:
  1. the architecture and the training loop are in _cgan_arch.py, the plumbing in
     _cgan_common.py; this file is the driver that the original kept inside an
     `if __name__ == '__main__':`.
  2. paths: the cGAN's data lives under ./data, the training table at
     ./data/cgan_train/hek293_train.csv, and everything the trainer writes goes under
     ./checks/cgan_hek293t -- SAVE_ROOT in _cgan_arch.py. The original read
     ./datasets/hek293_train.csv and wrote ./checkpoints, so its generator sat at
     ./checkpoints/check/deepseed_hek293_debug/net_G_49999.pth. Nothing inside the run changed: the
     50000 iterations, the 1000-iteration saving, the 0.8 split, the ten-seed augmentation of
     LoadData_Debug, the batch of 32, the five critics, l1_w 50 and the learning rate are the
     author's numbers, and the checkpoint by the name of the last iteration, 49999, is the one the
     generation step loads.
  3. the data preprocessing the original kept commented out under "## waste:" is kept here as a
     flag, BUILD_TRAIN_CSV. The training table is normally the file you copy into
     ./data/cgan_train; with the flag True it is rebuilt first from the two enhancer files beside
     it, realA the masked and realB the unmasked, which is what makes hek293_train.csv. Either way
     the table is 9316 sequences of 252 bp, uppercased as open_fa_withidx uppercases them.
  4. the files this needs are checked before the run rather than inside the library, so a missing
     one names itself.
  5. import joblib and the pdf/ps fonttype lines are the standing requirement for delivered files.

Training this wants a GPU: the generator's first layer alone is a Linear(252 * 4, 252 * 512), and
the code falls back to the CPU only in the sense that it will run there. With the pretrained
net_G_49999.pth at ./checks/cgan_hek293t/ there is nothing to do in this file at all; go straight to
R01_round1_prepare_lib_generate.py.

Running this file trains and writes the checkpoints and the cache; it prints a [report] line at the
start and at the end.
'''

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
