import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'

# Project layout: data in ./data, figures in ./figs (checkpoints, where used, in ./checks)
DATA_DIR = './data'
FIG_DIR = './figs'
FIG_STEM = 'S03_replicate_scatter'
DPI = 300

# The offset every log2 in this script takes: a level is log2(value + LOG_OFFSET). It is
# 1e-6, the same offset the limma statistic of the rest of the project uses, so this figure
# and the others share one convention. LOG_OFFSET_LABEL is what the axes print and has to
# move with it.
LOG_OFFSET = 1e-6
LOG_OFFSET_LABEL = '1e-6'

# 7_exp_rep1/rep2_(FUS)_TO_induced/uninduced: 4 experimental conditions, 2 biological replicates each
TABLE_FILES = {
    ('FUS-rtTA', 'Induced'):   ('7_exp_rep1_FUS_TO_induced.tsv',   '7_exp_rep2_FUS_TO_induced.tsv'),
    ('rtTA',     'Induced'):   ('7_exp_rep1_TO_induced.tsv',       '7_exp_rep2_TO_induced.tsv'),
    ('FUS-rtTA', 'Uninduced'): ('7_exp_rep1_FUS_TO_uninduced.tsv', '7_exp_rep2_FUS_TO_uninduced.tsv'),
    ('rtTA',     'Uninduced'): ('7_exp_rep1_TO_uninduced.tsv',     '7_exp_rep2_TO_uninduced.tsv'),
}
CONDITIONS = ['FUS-rtTA', 'rtTA']
TREATMENTS = [('Induced', '#74a892'), ('Uninduced', '#e5c185')]
SYNTAX_GROUPS = ['spacer', 'mutant', 'flanking']


def get_syntax_group(name):
    name = str(name).lower()
    if name.startswith('spacer_'):
        return 'spacer'
    elif name.startswith('mutant_'):
        return 'mutant'
    elif name.startswith('flanking_'):
        return 'flanking'
    elif name.startswith('positive_'):
        return 'positive'
    elif name.startswith('negative_'):
        return 'negative'
    else:
        return 'other'


def load_condition(rep1_file, rep2_file):
    df_rep1 = pd.read_csv(os.path.join(DATA_DIR, rep1_file), sep='\t', header=None)
    df_rep2 = pd.read_csv(os.path.join(DATA_DIR, rep2_file), sep='\t', header=None)
    df_rep1.columns = ['seqs', 'rep1', 'names']
    df_rep2.columns = ['seqs', 'rep2', 'names']
    df_merged = pd.merge(df_rep1, df_rep2, on=['names', 'seqs'], how='inner')

    # calibrate each biological replicate against its own negative control
    df_merged['rep1_raw'] = df_merged['rep1']
    df_merged['rep2_raw'] = df_merged['rep2']
    cali_rep1 = df_merged[df_merged['names'].str.contains('negative_random')]['rep1'].mean()
    cali_rep2 = df_merged[df_merged['names'].str.contains('negative_random')]['rep2'].mean()
    df_merged['rep1'] = df_merged['rep1'] / cali_rep1
    df_merged['rep2'] = df_merged['rep2'] / cali_rep2

    df_merged['syntax_group'] = df_merged['names'].apply(get_syntax_group)
    return df_merged


def pcc(values_x, values_y):
    data_x = np.log2(values_x + LOG_OFFSET)
    data_y = np.log2(values_y + LOG_OFFSET)
    mask = ~np.isnan(data_x) & ~np.isnan(data_y)
    r, _ = pearsonr(data_x[mask], data_y[mask])
    return r, data_x[mask], data_y[mask]


def scale_pcc(values_x, values_y):
    mask = ~np.isnan(values_x) & ~np.isnan(values_y)
    r, _ = pearsonr(values_x[mask], values_y[mask])
    return r


fig, axes = plt.subplots(2, 2, figsize=(12, 12))
records = []

for row_idx, (treatment, point_color) in enumerate(TREATMENTS):

    for col_idx, condition in enumerate(CONDITIONS):

        rep1_file, rep2_file = TABLE_FILES[(condition, treatment)]
        df_merged = load_condition(rep1_file, rep2_file)
        df_analysis = df_merged[df_merged['syntax_group'].isin(SYNTAX_GROUPS)].copy()

        r, data_x_clean, data_y_clean = pcc(df_analysis['rep1'].values, df_analysis['rep2'].values)

        ax = axes[row_idx, col_idx]
        ax.scatter(data_x_clean, data_y_clean, alpha=0.9, s=18, c=point_color, edgecolors='k',
                   linewidth=0.35, rasterized=True)

        lims = [0, 6]

        ax.plot(lims, lims, 'k--', alpha=0.45, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        text_str = f'PCC = {r:.3f}'

        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=13, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.8))
        ax.set_title(condition, fontsize=16, fontweight='bold')
        ax.set_xlabel(f'log2(rep1 + {LOG_OFFSET_LABEL})', fontsize=13)
        ax.set_ylabel(f'log2(rep2 + {LOG_OFFSET_LABEL})', fontsize=13)
        ax.tick_params(axis='both', labelsize=11)

        if col_idx == 0:
            ax.text(-0.32, 0.5, treatment, transform=ax.transAxes, rotation=90, fontsize=18,
                    fontweight='bold', va='center', ha='center')

        r_uncalibrated, _, _ = pcc(df_analysis['rep1_raw'].values, df_analysis['rep2_raw'].values)
        r_raw_scale = scale_pcc(df_analysis['rep1'].values, df_analysis['rep2'].values)
        r_all_rows, _, _ = pcc(df_merged['rep1'].values, df_merged['rep2'].values)

        records.append((condition, treatment, r, r_uncalibrated, r_raw_scale, r_all_rows,
                        data_x_clean.size, df_merged.shape[0]))

sns.despine()
plt.tight_layout()

os.makedirs(FIG_DIR, exist_ok=True)
for fmt in ('png', 'svg', 'pdf'):
    plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM}.{fmt}'), format=fmt, dpi=DPI, bbox_inches='tight')
plt.close()


# --- REPORT: A. Pearson's correlation between two biological replicates (N = 2), per condition ---

print()
print("[report] A. Pearson's correlation between two biological replicates (N = 2) from each experimental condition")
for condition, treatment, r, r_uncalibrated, r_raw_scale, r_all_rows, n_plot, n_merged in records:
    print(f"[report]   {condition} {treatment:<10s} PCC = {r:.4f}   "
          f"[negative-control calibrated, log2(x + {LOG_OFFSET_LABEL}), n = {n_plot}]")
    print(f"[report]   {'':<20s} PCC = {r_uncalibrated:.4f}   "
          f"[uncalibrated, log2(x + {LOG_OFFSET_LABEL}), n = {n_plot}]")
    print(f"[report]   {'':<20s} PCC = {r_raw_scale:.4f}   [negative-control calibrated, raw scale, n = {n_plot}]")
    print(f"[report]   {'':<20s} PCC = {r_all_rows:.4f}   "
          f"[negative-control calibrated, log2(x + {LOG_OFFSET_LABEL}), all {n_merged} merged rows incl. controls]")
