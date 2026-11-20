import os
import sys

import joblib
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.metrics import ndcg_score

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'

# The libraries of the architectures, which the project keeps in ./scripts
for _library in (os.path.dirname(os.path.abspath(__file__)), './scripts'):
    if _library and os.path.isdir(_library) and _library not in sys.path:
        sys.path.append(_library)

from lib_nnmodel import NNModel_language, open_fa, open_exp
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

# The project layout: the split in ./preds (written by S08_train_model.py), the weights in
# ./checks, the intermediates in ./preds, the figures in ./figs
SPLIT_DIR = './preds/S08_training_data/'
ARCHS_DIR = './checks/round1_archs/'          # the nine of S08_train_models.py
HOLDOUT_DIR = './checks/round1_holdout/'      # attnbilstm, the S08 model
PRED_DIR = './preds'
FIG_DIR = './figs'
DPI = 300

CSV_PATH = os.path.join(PRED_DIR, 'S08_bootstrap_model_comparison.csv')
FIG_STEM = 'S08_bootstrap_model_comparison'

# A csv on disk already holds the bootstrap these checkpoints give, so the figure can be redrawn
# without loading a single model
REUSE_CSV = False

BOOTSTRAP_N = 20
SAMPLE_SIZE = 100
BOOTSTRAP_SEED = 42

MODEL_TAGS = ["attnbilstm", "deepstarr2", "deepgoplus", "malinois", "legnet", "deepcross",
              "densenet", "denselstm", "resnethw", "resnetwy"]
TASKS = ["induced", "uninduced"]
TASK_TITLES = {"induced": "Induced", "uninduced": "Uninduced"}
METRICS = ["PCC", "NDCG"]

MODEL_CLASS_MAP = {
    "attnbilstm": AttnBilstm_language,
    "deepstarr2": DeepSTARR2_language,
    "deepgoplus": deepGoPlus_language,
    "malinois": Malinois_language,
    "legnet": LegNet_language,
    "deepcross": DeepCROSS_language,
    "densenet": DenseNet_language,
    "denselstm": DenseLSTM_language,
    "resnethw": ResNethw_language,
    "resnetwy": ResNetwy_language
}

# One colour for all ten architectures, the blue the author's own boxplot drew
BOX_COLOR =  "#4c72b0" # "#74a892" #


def checkpoint_dir(tag):
    '''Where S08_train_models.py put this architecture: attnbilstm is the S08 model.'''
    if tag == "attnbilstm":
        return HOLDOUT_DIR
    return f'{ARCHS_DIR}{tag}/'


def save_figure(stem):
    '''The three formats, at the one dpi the project writes its figures at, into ./figs.'''
    os.makedirs(FIG_DIR, exist_ok=True)
    for fmt in ('png', 'svg', 'pdf'):
        plt.savefig(os.path.join(FIG_DIR, f'{stem}.{fmt}'), format=fmt, dpi=DPI,
                    bbox_inches='tight')
    plt.close()


''' --Evaluation: 用bootstrap的结果来评估不同模型 -- '''

def bootstrap_results():
    results = []
    np.random.seed(BOOTSTRAP_SEED)

    for tag in MODEL_TAGS:
        print(f"\n===== Evaluating {tag} =====")
        checkpoint_dir_tag = checkpoint_dir(tag)

        for task_name in TASKS:
            print(f"  -> {task_name}")
            model = MODEL_CLASS_MAP[tag](model_name=task_name, length=252)
            model_path = os.path.join(checkpoint_dir_tag, task_name, "checkpoint.pth")
            seqs_file = os.path.join(SPLIT_DIR, "test_seqs.txt")
            label_file = os.path.join(SPLIT_DIR, f"test_{task_name}.txt")

            # the split, the labels and the weights all have to be where the paths say; say so
            # here rather than dying inside the library
            for _needed in (seqs_file, label_file, model_path):
                if not os.path.exists(_needed):
                    raise FileNotFoundError(
                        f'{_needed} is missing. S08_train_model.py writes the split into '
                        f'{SPLIT_DIR}; S08_train_models.py writes each architecture into '
                        f'{ARCHS_DIR}<tag>/<branch>/checkpoint.pth, and attnbilstm is '
                        f'{HOLDOUT_DIR}<branch>/checkpoint.pth')

            seqs = np.array(open_fa(seqs_file))
            expr = np.array(open_exp(label_file, operator="direct"))
            pred = np.array(
                model.predict_input(
                    model_path=model_path,
                    inputs=seqs, mode="data")
            )

            for b in range(BOOTSTRAP_N):
                idx = np.random.choice(len(expr), size=SAMPLE_SIZE, replace=True)
                y_true = expr[idx]
                y_pred = pred[idx]
                results.append({
                    "model": tag,
                    "task": task_name,
                    "bootstrap": b,
                    "PCC": pearsonr(y_true, y_pred)[0],
                    "NDCG": ndcg_score([y_true], [y_pred], k=min(50, SAMPLE_SIZE))
                })

            print(f'[report] {tag} / {task_name}: {len(expr)} held-out sequences, '
                  f'{BOOTSTRAP_N} bootstrap resamples of {SAMPLE_SIZE}')

    return pd.DataFrame(results)


os.makedirs(PRED_DIR, exist_ok=True)

if REUSE_CSV and os.path.exists(CSV_PATH):
    print(f'[reuse] reading the bootstrap from {CSV_PATH}, no checkpoint is loaded')
    df_results = pd.read_csv(CSV_PATH)
else:
    df_results = bootstrap_results()
    df_results.to_csv(CSV_PATH, index=False)
    print(f'[report] bootstrap written to {CSV_PATH}')


''' --图: 箱线图, 对比不同模型架构的性能影响 -- '''

# the sideways jitter of the strip comes from numpy's global state, so it is pinned here rather
# than left to whatever ran before: with REUSE_CSV the bootstrap has not advanced that state, and
# the same twenty numbers would otherwise come out in different places
np.random.seed(BOOTSTRAP_SEED)

fig, axes = plt.subplots(2, len(TASKS), figsize=(9, 8.5), sharex='col', sharey='row')
sns.set_style("white")

for row, metric in enumerate(METRICS):
    for col, task in enumerate(TASKS):
        ax = axes[row, col]
        df_plot = df_results[df_results["task"] == task]

        sns.boxplot(
            data=df_plot,
            x="model",
            y=metric,
            ax=ax,
            width=0.7,
            fliersize=1.5,
            color=BOX_COLOR
        )

        sns.stripplot(
            data=df_plot,
            x="model",
            y=metric,
            ax=ax,
            color="black",
            alpha=0.8,
            size=2.5
        )

        if row == 0:
            ax.set_title(TASK_TITLES[task], fontsize=16, fontweight='bold')
        ax.set_ylabel(metric if col == 0 else "", fontsize=16)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)
        # ax.grid(True, alpha=0.3)

plt.tight_layout()
save_figure(FIG_STEM)
print(f'[report] figure written to {os.path.join(FIG_DIR, FIG_STEM)}.png/.svg/.pdf')


''' --mean over the resamples: one line per model, branch by branch -- '''

print('\n===== mean over the %d resamples =====' % BOOTSTRAP_N)
for task in TASKS:
    df_task = df_results[df_results["task"] == task]
    print(f'\n{TASK_TITLES[task]}')

    for tag in MODEL_TAGS:
        df_model = df_task[df_task["model"] == tag]
        fields = []
        for metric in METRICS:
            mean = df_model[metric].mean()
            std = df_model[metric].std()
            fields.append(f'{metric} {mean:.4f} +/- {std:.4f}')
        print(f'  {tag:<11} ' + '   '.join(fields))
