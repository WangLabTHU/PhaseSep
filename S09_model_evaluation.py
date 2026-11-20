import os
import sys

import joblib
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import pearsonr

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'

# The library of the model: _model beside this file (refs_dqx/model.py is byte for byte
# refs_dqx/scripts/lib_attnbilstm.py, and both take their helpers -- open_fa, open_exp, seq2onehot,
# EarlyStopping -- from utils.py). ./scripts and the project root are looked in too, so the file
# also runs from inside the reference tree.
for _library in (os.path.dirname(os.path.abspath(__file__)), './scripts', './'):
    if _library and os.path.isdir(_library) and _library not in sys.path:
        sys.path.append(_library)

try:
    from _model import AttnBilstm_language, open_fa, open_exp
except ImportError:
    from model import AttnBilstm_language, open_fa, open_exp

# The project layout: the tables in ./data, the split in ./preds, the weights in ./checks,
# the figures in ./figs
DATA_DIR = './data'
FIG_DIR = './figs'
DPI = 300

# What S09_train_model.py writes: the held-out tenth here, the three branches under
# ./checks/<model>/. The library builds a branch's directory as savepath + model_name, so that
# one keeps its trailing slash, and so does the split directory for symmetry.
SPLIT_DIR = './preds/S08_training_data/'
MODEL_NAME = 'round1_holdout'
CHECKPOINT_DIR = './checks/' + MODEL_NAME + '/'

FILE_FUS_INDUCED = '7_exp_mean_FUS_TO_induced.tsv'
FILE_FUS_UNINDUCED = '7_exp_mean_FUS_TO_uninduced.tsv'

FIG_STEM = {'induced': 'S09_regplot_induced', 'uninduced': 'S09_regplot_uninduced'}
FIG_STEM_NDCG = 'S09_ndcg_curve'


def save_figure(stem):
    '''The three formats, at the one dpi the project writes its figures at, into ./figs.'''
    os.makedirs(FIG_DIR, exist_ok=True)
    for fmt in ('png', 'svg', 'pdf'):
        plt.savefig(os.path.join(FIG_DIR, f'{stem}.{fmt}'), format=fmt, dpi=DPI,
                    bbox_inches='tight')
    plt.close()


def plot_reg(label_real, label_pred, stem, c1="indianred"):
    from scipy import stats

    def r(x, y):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return r_value

    r_test = r(label_real, label_pred)
    testdata = pd.DataFrame({'pred': label_pred, 'label': label_real})

    sns.set_style("white")
    font = {'size' : 18}
    matplotlib.rc('font', **font)
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    fig, ax = plt.subplots(figsize = (6,6), dpi = 300)

    ax.scatter(
        testdata['label'],
        testdata['pred'],
        alpha=0.9,
        s=18,
        c=c1,
        edgecolors='k',
        linewidth=0.35
    )

    lim_min = min(testdata['label'].min(), testdata['pred'].min())
    lim_max = max(testdata['label'].max(), testdata['pred'].max())
    lims = [lim_min, lim_max]

    ax.plot(
        lims,
        lims,
        'k--',
        alpha=0.45,
        zorder=0
    )

    ax.set_xlim(lims)
    ax.set_ylim(lims)

    text_str = f'PCC = {r_test:.3f}'
    ax.text(
        0.05,
        0.95,
        text_str,
        transform=ax.transAxes,
        fontsize=12,
        fontweight='bold',
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            edgecolor='none',
            alpha=0.8
        )
    )

    ax.set_xlabel('Real Fold Change', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)

    sns.despine()

    save_figure(stem)
    return r_test


def evaluate_branch(task_name, base_dir, checkpoint_base, length=252, color="mediumpurple"):
    model = AttnBilstm_language(model_name=task_name, length=length)

    model_path = os.path.join(checkpoint_base, task_name, "checkpoint.pth")
    seqs_file = os.path.join(base_dir, "test_seqs.txt")
    label_file = os.path.join(base_dir, f"test_{task_name}.txt")

    # the split, the labels and the weights all have to be where the paths say, and they are
    # written by S09_train_model.py, so say what is missing rather than dying inside the library
    for needed in (seqs_file, label_file, model_path):
        if not os.path.exists(needed):
            raise FileNotFoundError(
                f'{needed} is missing. S09_train_model.py writes the split into {SPLIT_DIR} and '
                f'each branch into {CHECKPOINT_DIR}<branch>/checkpoint.pth')

    seqs = open_fa(seqs_file)
    expr = open_exp(label_file, operator="direct")
    pred = model.predict_input(model_path=model_path, inputs=seqs, mode="data")

    pcc = plot_reg(expr, pred, FIG_STEM[task_name], c1=color)
    print(f'[report] {task_name}: {len(expr)} held-out sequences, PCC = {pcc:.3f}')

    return expr, pred


def ndcg_at_k(y_true, y_pred, k=10):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) < k:
        k = len(y_true)

    top_k_pred_idx = np.argsort(y_pred)[::-1][:k]
    top_k_true_idx = np.argsort(y_true)[::-1][:k]

    rel = y_true[top_k_pred_idx]
    ideal_rel = y_true[top_k_true_idx]

    dcg = np.sum((2**rel - 1) / np.log2(np.arange(2, k + 2)))
    idcg = np.sum((2**ideal_rel - 1) / np.log2(np.arange(2, k + 2)))

    return dcg / idcg if idcg != 0 else 0.0


def evaluate_ndcg_scores(true_dict, pred_dict, ks=None, stem=None):
    if ks is None:
        ks = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200]

    color_map = {
        "induced": "#74a892",
        "uninduced": "#e5c185"
    }

    all_results = {}

    print("\n==== NDCG Evaluation ====")
    for branch in true_dict:
        y_true = true_dict[branch]
        y_pred = pred_dict[branch]
        ndcg_scores = []

        print(f"\n{branch.upper()}:")
        for k in ks:
            score = ndcg_at_k(y_true, y_pred, k)
            ndcg_scores.append(score)
            print(f"  NDCG@{k:<3}: {score:.4f}")
        all_results[branch] = ndcg_scores

    sns.set_style("white")
    font = {'size' : 14}
    matplotlib.rc('font', **font)
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

    fig, ax = plt.subplots(figsize=(6, 6))
    for branch, scores in all_results.items():
        color = color_map.get(branch, "gray")
        ax.plot(ks, scores, marker='o', label=branch, color=color, linewidth=2, markersize=6, markeredgecolor='k', markeredgewidth=0.5)
        for x, y in zip(ks[-6:], scores[-6:]):
            ax.text(x, y + 0.015, f"{y:.3f}", fontsize=10, ha='center', va='bottom')

    ax.set_xlabel("K", fontsize=14)
    ax.set_ylabel("NDCG@K", fontsize=14)
    ax.set_title("NDCG@K Curve", fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=12)

    sns.despine()
    plt.tight_layout()

    if stem is not None:
        save_figure(stem)
    else:
        plt.close()


''' regplot / ndcg curves '''

print(f'[report] {MODEL_NAME}: split in {SPLIT_DIR}, checkpoints in {CHECKPOINT_DIR}, '
      f'tables in {DATA_DIR}, figures in {FIG_DIR}')

df_induced = pd.read_csv(os.path.join(DATA_DIR, FILE_FUS_INDUCED), sep="\t", header=None)
df_uninduced = pd.read_csv(os.path.join(DATA_DIR, FILE_FUS_UNINDUCED), sep="\t", header=None)
df_induced.columns = ["seqs", "induced_mpra_means", "names"]
df_uninduced.columns = ["seqs", "uninduced_mpra_means", "names"]

df_merged = pd.merge(df_induced, df_uninduced, on=['names', 'seqs'], how='inner')
df_merged = df_merged[["seqs", "names", "induced_mpra_means", "uninduced_mpra_means"]]
df_merged["lfc"] = df_merged["induced_mpra_means"] / df_merged["uninduced_mpra_means"]

base_dir = SPLIT_DIR
checkpoint_dir = CHECKPOINT_DIR

true_induced, pred_induced = evaluate_branch("induced", base_dir, checkpoint_dir, color="#74a892")
true_uninduced, pred_uninduced = evaluate_branch("uninduced", base_dir, checkpoint_dir, color="#e5c185")

true_dict = { "induced": true_induced, "uninduced": true_uninduced}
pred_dict = {"induced": pred_induced, "uninduced": pred_uninduced}
evaluate_ndcg_scores(true_dict, pred_dict, stem=FIG_STEM_NDCG)
