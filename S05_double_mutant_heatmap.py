import os
from functools import reduce

import joblib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Project layout: data in ./data, figures in ./figs.
DATA_DIR = './data'
FIG_DIR = './figs'
FIG_STEM = 'S05_double_mutant_heatmap'
DPI = 600

REP_FILES = {
    'induced_rep1': '7_exp_rep1_FUS_TO_induced.tsv',
    'induced_rep2': '7_exp_rep2_FUS_TO_induced.tsv',
    'uninduced_rep1': '7_exp_rep1_FUS_TO_uninduced.tsv',
    'uninduced_rep2': '7_exp_rep2_FUS_TO_uninduced.tsv',
}
WT_NAME = 'positive_wildtype_clontech_7tetO_rep0'
PSEUDOCOUNT = 1

# --- NATURE PRESS GLOBAL FONT CONFIGURATION (as in the reference) -------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams.update({
    'font.size': 5,
    'axes.labelsize': 5,
    'axes.titlesize': 5,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'lines.linewidth': 0.25,
    'axes.linewidth': 0.25,
    'patch.linewidth': 0.25,
    'svg.fonttype': 'none',
})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

os.makedirs(FIG_DIR, exist_ok=True)


# --- STEP 3.0: CALIBRATE THE FOUR ROUND-1 REPLICATE TABLES --------------------

def read_rep_table(filename, column):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{filename} is missing from {DATA_DIR}; this is one of the four round-1 "
            "replicate tables the double-mutant set is built from.")
    df = pd.read_csv(path, sep='\t', header=None)
    if df.shape[1] != 3:
        raise ValueError(
            f"{filename} has {df.shape[1]} columns; expected the three columns "
            "'seq, value, name'.")
    df.columns = ['seqs', column, 'names']
    return df[['seqs', 'names', column]]


rep_frames = [read_rep_table(filename, column) for column, filename in REP_FILES.items()]
merged = reduce(lambda left, right: pd.merge(left, right, on=['names', 'seqs'],
                                             how='inner'), rep_frames)
if merged.empty:
    raise ValueError(
        "the four replicate tables share no (names, seqs) rows; the reference merged "
        "them with an inner join and would have drawn an empty figure.")

rep_columns = list(REP_FILES)
cali = merged.loc[merged['names'].str.contains('negative_random'), rep_columns].mean()
for column in rep_columns:
    if not np.isfinite(cali[column]) or cali[column] <= 0:
        raise ValueError(
            f"the negative_random rows give a calibration of {cali[column]} for "
            f"{column}; every calibrated value would be NaN or sign-flipped.")

for column in rep_columns:
    merged[column] = merged[column] / cali[column]

merged['induced_mean'] = merged[['induced_rep1', 'induced_rep2']].mean(axis=1)
merged['uninduced_mean'] = merged[['uninduced_rep1', 'uninduced_rep2']].mean(axis=1)
merged['lfc'] = np.log2((merged['induced_mean'] + PSEUDOCOUNT) /
                        (merged['uninduced_mean'] + PSEUDOCOUNT))
merged = merged[['seqs', 'names', 'induced_rep1', 'induced_rep2', 'induced_mean',
                 'uninduced_rep1', 'uninduced_rep2', 'uninduced_mean', 'lfc']]

# --- STEP 3.0: DOUBLE-MUTANT SET, PALINDROMIC SET AND WT ----------------------

wt_set = merged[merged['names'] == WT_NAME]
if len(wt_set) != 1:
    raise ValueError(
        f"expected exactly one {WT_NAME!r} row in the calibrated table, found "
        f"{len(wt_set)}; the wild type is the last row of the reference table and the "
        "L2FC is measured against it.")

mutant_set = merged[merged['names'].str.contains(r'mutant_O2', regex=True)]
palindromic_set = merged[merged['names'].str.contains(r'mutant_P_O2', regex=True)]
if mutant_set.empty:
    raise ValueError("no row name contains 'mutant_O2'; there is no double-mutant set.")


def parse_mutant_names(frame, expected_fields):
    '''Split group_motif_1st_mutant_2nd_mutant (and the palindromic variant, which
    carries one more field) and pull the two positions and bases out.'''
    label = frame['names'].str.split('_', expand=True)
    if label.shape[1] != len(expected_fields):
        raise ValueError(
            f"the {len(frame)} selected names do not split into the "
            f"{len(expected_fields)} fields {expected_fields} (got {label.shape[1]}); "
            "the reference assigned the column names unconditionally")
    label.columns = expected_fields
    # the extracted digits arrive as object strings; older pandas cannot cast them
    # straight to the nullable Int64 dtype, so go through pd.to_numeric first
    label['1st_position'] = pd.to_numeric(
        label['1st_mutant'].str.extract(r'(\d+)', expand=False), errors='coerce')
    label['1st_mutated_base'] = label['1st_mutant'].str.extract(r'([A-Za-z]+)', expand=False)
    label['2nd_position'] = pd.to_numeric(
        label['2nd_mutant'].str.extract(r'(\d+)', expand=False), errors='coerce')
    label['2nd_mutated_base'] = label['2nd_mutant'].str.extract(r'([A-Za-z]+)', expand=False)
    return pd.concat([frame, label[['1st_position', '1st_mutated_base',
                                    '2nd_position', '2nd_mutated_base']]], axis=1)


mutant_label = parse_mutant_names(
    mutant_set, ['group', 'motif_name', '1st_mutant', '2nd_mutant'])
palindromic_label = parse_mutant_names(
    palindromic_set, ['group', 'motif_name', 'type', '1st_mutant', '2nd_mutant'])

# the reference counted non-null cells over its whole 13-column table (nine data columns
# plus the four parsed fields) and called 13 a double mutant; counting the four parsed
# fields says the same thing without depending on how many data columns the table
# happens to carry, which is how the single-mutant count was fixed in F01e
parsed_fields = ['1st_position', '1st_mutated_base', '2nd_position', '2nd_mutated_base']
double_mutant = mutant_label[
    mutant_label[parsed_fields].notnull().sum(axis=1) == len(parsed_fields)].copy()
if double_mutant.empty:
    raise ValueError(
        "no mutant_O2 name carries two complete mutation fields; the reference table "
        "would have had no double mutants to draw.")

double_mutant_data = pd.concat([double_mutant, palindromic_label], ignore_index=True)
double_mutant_data = pd.concat([double_mutant_data, wt_set], ignore_index=True)


# --- STEP 3.3: THE HEATMAPS (supp_1_double_mutant_plot_V2.py) -----------------

df = double_mutant_data

# The wild type is the very last row of the table, as in the reference.
wt_row = df.iloc[-1]
wt_ind_mean = np.array([wt_row['induced_rep1'], wt_row['induced_rep2']]).mean()
wt_unind_mean = np.array([wt_row['uninduced_rep1'], wt_row['uninduced_rep2']]).mean()

df_variants = df.iloc[:-1].copy()
df_variants['induced_mean'] = df_variants[['induced_rep1', 'induced_rep2']].mean(axis=1)
df_variants['uninduced_mean'] = df_variants[['uninduced_rep1', 'uninduced_rep2']].mean(axis=1)
df_variants['L2FC_induced'] = np.log2(df_variants['induced_mean'] / wt_ind_mean)
df_variants['L2FC_uninduced'] = np.log2(df_variants['uninduced_mean'] / wt_unind_mean)

pos_plot_df = df_variants.groupby(['1st_position', '2nd_position']).agg({
    'L2FC_induced': 'mean',
    'L2FC_uninduced': 'mean',
}).reset_index()

max_pos = int(max(pos_plot_df['1st_position'].max(), pos_plot_df['2nd_position'].max()))

matrix_induced = np.full((max_pos, max_pos), np.nan)
matrix_uninduced = np.full((max_pos, max_pos), np.nan)

for _, row in pos_plot_df.iterrows():
    x_idx = int(row['2nd_position']) - 1
    y_idx = int(row['1st_position']) - 1
    matrix_induced[y_idx, x_idx] = row['L2FC_induced']
    matrix_uninduced[y_idx, x_idx] = row['L2FC_uninduced']

# --- 4. Horizontal Page Layout Setup ---
width_inch = 136 / 25.4
height_inch = 55 / 25.4

fig = plt.figure(figsize=(width_inch, height_inch), layout='constrained')

gs_master = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 0.15, 0.05], wspace=0.1)

ax_ind = fig.add_subplot(gs_master[0, 0])
ax_unind = fig.add_subplot(gs_master[0, 1], sharex=ax_ind, sharey=ax_ind)
cbar_ax = fig.add_subplot(gs_master[0, 3])

# --- 5. Unified Colormap Setup ---
cmap_shared = sns.diverging_palette(10, 130, sep=10, as_cmap=True)
cmap_shared.set_bad(color='white')
norm_shared = plt.Normalize(vmin=-3.0, vmax=1.0)

# --- 6. Render Panels using Seaborn Heatmap with masks ---
# the cells are parted by white seams, not by the black ones the V2 code draws: the
# assembled artwork (supp_5_horizontal_double_mutant.tif) has no black mesh anywhere
mask_ind = np.isnan(matrix_induced)
mask_unind = np.isnan(matrix_uninduced)

sns.heatmap(
    matrix_induced,
    ax=ax_ind,
    cmap=cmap_shared,
    norm=norm_shared,
    cbar=False,
    mask=mask_ind,
    linewidths=0.25,
    linecolor='white',
    square=True,
)
ax_ind.set_title("Induced Activity", fontweight='bold', fontsize=5.5, color='#012f48', pad=3)

sns.heatmap(
    matrix_uninduced,
    ax=ax_unind,
    cmap=cmap_shared,
    norm=norm_shared,
    cbar=False,
    mask=mask_unind,
    linewidths=0.25,
    linecolor='white',
    square=True,
)
ax_unind.set_title("Leakiness", fontweight='bold', fontsize=5.5, color='#669aba', pad=3)

# Axis Configurations
for ax in [ax_ind, ax_unind]:
    ax.set_xticks(np.arange(0.5, max_pos, 2))
    ax.set_xticklabels(range(1, max_pos + 1, 2))
    ax.set_yticks(np.arange(0.5, max_pos, 2))
    ax.set_yticklabels(range(1, max_pos + 1, 2))
    ax.tick_params(axis='both', which='major', length=1.5, width=0.25, pad=1)

    # Move Y-axis ticks and labels to the right side
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # seaborn's heatmap turns every spine off; the artwork keeps the frame on the two
    # sides that carry the ticks (bottom and right) and leaves the other two open
    for side in ('top', 'left'):
        ax.spines[side].set_visible(False)
    for side in ('bottom', 'right'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.25)
        ax.spines[side].set_color('black')

# Common axis labels
ax_ind.set_xlabel('1st Mutation Position (bp)', fontweight='bold', fontsize=5.5)
ax_unind.set_xlabel('1st Mutation Position (bp)', fontweight='bold', fontsize=5.5)
ax_unind.set_ylabel('2nd Mutation Position (bp)', fontweight='bold', fontsize=5.5)

ax_ind.tick_params(axis='y', labelright=False)

# The reference artwork (supp_5_horizontal_double_mutant.tif) runs the second mutation
# position downwards and the first one upwards, 19 at the top; the V2 code draws them
# the other way round, so the shared axis is flipped here to match the artwork.
ax_ind.invert_yaxis()

# --- 7. Colorbar Setup ---
sm = plt.cm.ScalarMappable(cmap=cmap_shared, norm=norm_shared)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=5, length=1.5, width=0.25, pad=1)
cbar.outline.set_linewidth(0.25)
cbar.set_label('Relative mean calibrated MPRA activity (vs. WT)', fontweight='bold',
               labelpad=4, fontsize=5)

# --- 8. Save outputs ---
# dpi is given to the png only: on the vector formats it only rounds the tight bbox,
# and at 600 it moved the colour bar's right edge by one pixel against the reference
plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM}.pdf'), format='pdf',
            bbox_inches='tight', transparent=False)
plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM}.svg'), format='svg',
            bbox_inches='tight', transparent=False)
plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM}.png'), format='png', dpi=DPI,
            bbox_inches='tight', transparent=False)
plt.close()


# --- REPORT ------------------------------------------------------------------
print("[report] negative-control calibration (per replicate): "
      + ", ".join(f"{column} {cali[column]:.4f}" for column in rep_columns))
print(f"[report] double-mutant table: {len(double_mutant)} double mutants + "
      f"{len(palindromic_label)} palindromic + 1 WT = {len(double_mutant_data)} rows")
print(f"[report] position pairs {len(pos_plot_df)}, {max_pos} x {max_pos} grid, "
      f"{int(np.isfinite(matrix_induced).sum())} filled cells")
print(f"[report] WT means (calibrated): induced {wt_ind_mean:.4f}, "
      f"uninduced {wt_unind_mean:.4f}")
print(f"[report] L2FC ranges: induced [{np.nanmin(matrix_induced):.3f}, "
      f"{np.nanmax(matrix_induced):.3f}], uninduced [{np.nanmin(matrix_uninduced):.3f}, "
      f"{np.nanmax(matrix_uninduced):.3f}]")
