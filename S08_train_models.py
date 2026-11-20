import os
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# The libraries of the architectures, which the project keeps in ./scripts
for _library in (os.path.dirname(os.path.abspath(__file__)), './scripts'):
    if _library and os.path.isdir(_library) and _library not in sys.path:
        sys.path.append(_library)

from lib_nnmodel import NNModel_language
from lib_attnbilstm import AttnBilstm_language
from lib_deepstarr2 import DeepSTARR2_language
from lib_deepgoplus import deepGoPlus_language
from lib_malinois import Malinois_language
from lib_legnet import LegNet_language
from lib_deepcross import DeepCROSS_language
from lib_densenet import DenseNet_language
from lib_denselstm import DenseLSTM_language
from lib_resnethw import ResNethw_language
from lib_resnetwy import ResNetwy_language

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# The project layout: the split under ./preds (written by S08_train_model.py), the weights under
# ./checks
SPLIT_DIR = './preds/S08_training_data/'
ARCHS_DIR = './checks/round1_archs/'

# The architectures the figure compares, minus attnbilstm: its weights are the S08 model's, in
# ./checks/round1_holdout/, and S08_train_model.py is what trains them. Add "attnbilstm" to this
# list only if a second AttnBiLSTM fit is really wanted.
TAGS = ["deepstarr2", "deepgoplus", "malinois", "legnet", "deepcross",
        "densenet", "denselstm", "resnethw", "resnetwy"]

''' --使用的模型: 来自ALDE的一系列模型 -- '''

def train_branch(task_name, seqs_file, labels_file, save_dir, length=252, epoch=400, patience=50, tag="nnmodel"):
    os.makedirs(save_dir, exist_ok=True)
    if tag == "nnmodel":
        model = NNModel_language(length=length, epoch=epoch, patience=patience, model_name=task_name)
    elif tag == "attnbilstm":
        model = AttnBilstm_language(length=length, epoch=epoch, patience=patience, model_name=task_name)
    elif tag == "deepstarr2":
        model = DeepSTARR2_language(length=length, epoch=epoch, patience=patience, model_name=task_name)
    elif tag == "deepgoplus":
        model = deepGoPlus_language(length=length, epoch=epoch, patience=patience, model_name=task_name)
    elif tag == "malinois":
        model = Malinois_language(length=length, epoch=epoch, patience=patience, model_name=task_name)
    elif tag == "legnet":
        model = LegNet_language(length=length, epoch=epoch, patience=patience, model_name=task_name)
    elif tag == "deepcross":
        model = DeepCROSS_language(length=length, epoch=epoch, patience=patience, model_name=task_name)
    elif tag == "densenet":
        model = DenseNet_language(length=length, epoch=epoch, patience=patience, model_name=task_name)
    elif tag == "denselstm":
        model = DenseLSTM_language(length=length, epoch=epoch, patience=patience, model_name=task_name)
    # elif tag == "denseconlstm":
    #     model = DenseConnectedLSTM_language(length=length, epoch=epoch, patience=patience, model_name=task_name)
    elif tag == "resnethw":
        model = ResNethw_language(length=length, epoch=epoch, patience=patience, model_name=task_name)
    elif tag == "resnetwy":
        model = ResNetwy_language(length=length, epoch=epoch, patience=patience, model_name=task_name)
    model.train(dataset=seqs_file, labels=labels_file, savepath=save_dir)

seqs_file = os.path.join(SPLIT_DIR, 'train_seqs.txt')

# the split has to be on disk: it comes from S08_train_model.py, and without it every branch
# would fail inside the library rather than here
for _needed in (seqs_file, os.path.join(SPLIT_DIR, 'train_induced.txt'),
                os.path.join(SPLIT_DIR, 'train_uninduced.txt'),
                os.path.join(SPLIT_DIR, 'train_lfc.txt')):
    if not os.path.exists(_needed):
        raise FileNotFoundError(
            f'{_needed} is missing. S08_train_model.py writes the round-1 holdout split into '
            f'{SPLIT_DIR}; run it first, or point SPLIT_DIR at the split on disk')

for tag in TAGS:
    # train_branch makes the directory, so it does not have to exist first
    checkpoint_dir = f'{ARCHS_DIR}{tag}/'
    print(f'[report] {tag}: training the three branches into {checkpoint_dir}')

    train_branch(
        task_name="induced",
        seqs_file=seqs_file,
        labels_file=os.path.join(SPLIT_DIR, 'train_induced.txt'),
        save_dir=checkpoint_dir,
        tag=tag
    )

    train_branch(
        task_name="uninduced",
        seqs_file=seqs_file,
        labels_file=os.path.join(SPLIT_DIR, 'train_uninduced.txt'),
        save_dir=checkpoint_dir,
        tag=tag
    )

    train_branch(
        task_name="lfc",
        seqs_file=seqs_file,
        labels_file=os.path.join(SPLIT_DIR, 'train_lfc.txt'),
        save_dir=checkpoint_dir,
        tag=tag
    )
