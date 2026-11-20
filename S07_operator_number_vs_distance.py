import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project layout: data in ./data, figures in ./figs.
DATA_DIR = './data'
FIG_DIR = './figs'
FIG_STEM = 'S07_operator_number_vs_distance'
DPI = 600

# The four round-1 mean tables under the renamed project file names.
FILE_FUS_INDUCED = '7_exp_mean_FUS_TO_induced.tsv'
FILE_FUS_UNINDUCED = '7_exp_mean_FUS_TO_uninduced.tsv'
FILE_TO_INDUCED = '7_exp_mean_TO_induced.tsv'
FILE_TO_UNINDUCED = '7_exp_mean_TO_uninduced.tsv'

# The standard 7-tetO TRE of each arm, calibrated like the rest of the table (they are
# the means of the positive_wildtype_clontech_7tetO rows; the [report] prints that
# comparison). The reference hard-codes them, and they are kept hard-coded so the dashed
# lines cannot move with a change anywhere else in the table.
FUS_WT_EXPRESSION = 35.529396561119405
FUS_WT_LEAKINESS = 4.847596517853063
FUS_WT_LFC = 7.018707879469652
TETR_WT_EXPRESSION = 28.8234520256371
TETR_WT_LEAKINESS = 7.416776812285397
TETR_WT_LFC = 3.8925304731359915

GAP_LIST = [17, 20, 21]

# --- NATURE PRESS GLOBAL FONT CONFIGURATION (as in the reference) -------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams.update({
    'font.size': 5,
    'axes.labelsize': 5,
    'axes.titlesize': 5,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'lines.linewidth': 0.5,  # cleaner line weight for small figures
})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

os.makedirs(FIG_DIR, exist_ok=True)


# --- THE degseq_induced.csv TABLE, BUILT IN MEMORY ----------------------------

def read_mean_table(filename, value_column, label):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{filename} is missing from {DATA_DIR} (the {label} mean table)")
    df = pd.read_csv(path, sep='\t', header=None)
    if df.shape[1] != 3:
        raise ValueError(
            f"{filename} has {df.shape[1]} columns; expected the three columns "
            "'seq, value, name'.")
    df.columns = ['seqs', value_column, 'names']
    return df[['seqs', 'names', value_column]]


def build_arm(label, induced_file, uninduced_file):
    '''One arm of the degseq table: the two mean tables merged, the fold change of each
    sequence, and all three columns calibrated by the negative controls of that column,
    exactly as refs_dqx/round1_analyze_degseq.py does.'''
    induced = read_mean_table(induced_file, 'induced_mpra_means', label)
    uninduced = read_mean_table(uninduced_file, 'uninduced_mpra_means', label)
    merged = pd.merge(induced, uninduced, on=['names', 'seqs'], how='inner')
    if merged.empty:
        raise ValueError(
            f"the {label} induced and uninduced mean tables share no (names, seqs) "
            "rows; the reference's merge would have been empty as well")
    merged = merged[['seqs', 'names', 'induced_mpra_means', 'uninduced_mpra_means']]
    merged['lfc'] = merged['induced_mpra_means'] / merged['uninduced_mpra_means']

    columns = ['induced_mpra_means', 'uninduced_mpra_means', 'lfc']
    cali = merged.loc[merged['names'].str.contains('negative_random'), columns].mean()
    # the original divided by these without checking: one missing negative_random row,
    # or a calibration of zero, turns every calibrated value into NaN
    for column in columns:
        if not np.isfinite(cali[column]) or cali[column] <= 0:
            raise ValueError(
                f"the negative_random rows of the {label} arm give a calibration of "
                f"{cali[column]} for {column}, so the whole arm would come out as NaN")
        merged[column] = merged[column] / cali[column]
    return merged, cali


df_fus, cali_fus = build_arm('FUS-rtTA', FILE_FUS_INDUCED, FILE_FUS_UNINDUCED)
df_tetr, cali_tetr = build_arm('rtTA', FILE_TO_INDUCED, FILE_TO_UNINDUCED)

# the table refs_dqx/round1_analyze_degseq.py writes as degseq_induced.csv (apart from
# its A / M / Z_score / p_value columns, which this figure does not use)
data = pd.merge(df_fus, df_tetr, on=['seqs', 'names'], suffixes=('_FUS', '_tetR'))

# --- THE FIGURE (supp_7_operator_number_vs_distance.py) -----------------------


def plot_styled_errorbar(ax, x, y, yerr, fmt, color, label):
    eb = ax.errorbar(x, y, yerr=yerr, fmt=fmt, markersize=3, linewidth=0.5,
                     capsize=2, color=color, label=label)
    for barline in eb[2]:
        barline.set_linewidth(0.25)
    for cap in eb[1]:
        cap.set_markeredgewidth(0.25)


# --- EXACT CANVAS DIMENSIONS (double-column width, ~180 mm x 110 mm) ---
width_inch = 175 / 25.4
height_inch = 95 / 25.4
fig, axes = plt.subplots(3, 3, figsize=(width_inch, height_inch), sharex=True)

for i, gap_num in enumerate(GAP_LIST):
    pattern = rf'(?=.*{gap_num}gap)(?=.*tetO)'

    motif_number = data[data['names'].str.contains(pattern, regex=True)]
    if motif_number.empty:
        raise ValueError(
            f"no row name contains both '{gap_num}gap' and 'tetO'; the {gap_num} bp row "
            "of the reference figure would have been empty")
    label = motif_number['names'].str.split('_', expand=True)
    if label.shape[1] != 5:
        raise ValueError(
            f"the {gap_num} bp names do not split into the five fields "
            f"'group_motif number_gap_flanking_replicate' (got {label.shape[1]}); the "
            "reference assigned five column names unconditionally")
    label.columns = ['group', 'motif number', 'gap', 'flanking', 'replicate']
    motif_number_label = pd.concat([motif_number, label], axis=1)
    motif_number_label['motif_num'] = (
        motif_number_label['motif number'].str.extract(r'(\d+)').astype(int))

    group_tet_on_expression = motif_number_label.groupby('motif_num')['induced_mpra_means_tetR'].agg(['mean', 'std']).reset_index()
    group_tet_on_leakiness = motif_number_label.groupby('motif_num')['uninduced_mpra_means_tetR'].agg(['mean', 'std']).reset_index()
    group_tet_on_lfc = motif_number_label.groupby('motif_num')['lfc_tetR'].agg(['mean', 'std']).reset_index()
    group_fus_tet_on_expression = motif_number_label.groupby('motif_num')['induced_mpra_means_FUS'].agg(['mean', 'std']).reset_index()
    group_fus_tet_on_leakiness = motif_number_label.groupby('motif_num')['uninduced_mpra_means_FUS'].agg(['mean', 'std']).reset_index()
    group_fus_tet_on_lfc = motif_number_label.groupby('motif_num')['lfc_FUS'].agg(['mean', 'std']).reset_index()

    # Column 1: Induced Activity
    ax = axes[i, 0]
    plot_styled_errorbar(ax, group_fus_tet_on_expression['motif_num'], group_fus_tet_on_expression['mean'], group_fus_tet_on_expression['std'], '-x', 'red', 'FUS-rtTA' if i == 0 else "")
    plot_styled_errorbar(ax, group_tet_on_expression['motif_num'], group_tet_on_expression['mean'], group_tet_on_expression['std'], '-o', 'blue', 'rtTA' if i == 0 else "")
    ax.axhline(y=FUS_WT_EXPRESSION, color='red', linestyle='--', label='Standard TRE (with FUS-rtTA)', linewidth=0.5, alpha=0.6)
    ax.axhline(y=TETR_WT_EXPRESSION, color='blue', linestyle='--', label='Standard TRE (with rtTA)', linewidth=0.5, alpha=0.6)
    ax.set_ylabel('Mean Induced Activity', fontsize=5)
    ax.set_xlim(1, 8)
    ax.set_ylim(0, 50)
    ax.tick_params(axis='both', which='major', length=1, width=0.25, labelsize=5, pad=1)
    ax.text(-0.25, 0.5, f'distance = {gap_num} bp', va='center', ha='left', rotation=90, fontsize=6, transform=ax.transAxes)

    # Column 2: Leakiness
    ax = axes[i, 1]
    plot_styled_errorbar(ax, group_fus_tet_on_leakiness['motif_num'], group_fus_tet_on_leakiness['mean'], group_fus_tet_on_leakiness['std'], '-x', 'red', 'FUS-rtTA' if i == 0 else "")
    plot_styled_errorbar(ax, group_tet_on_leakiness['motif_num'], group_tet_on_leakiness['mean'], group_tet_on_leakiness['std'], '-o', 'blue', 'rtTA' if i == 0 else "")
    ax.axhline(y=FUS_WT_LEAKINESS, color='red', linestyle='--', label='Standard TRE (with FUS-rtTA)', linewidth=0.5, alpha=0.6)
    ax.axhline(y=TETR_WT_LEAKINESS, color='blue', linestyle='--', label='Standard TRE (with rtTA)', linewidth=0.5, alpha=0.6)
    ax.set_ylabel('Mean Leakiness', fontsize=5)
    ax.set_xlim(1, 8)
    ax.set_ylim(0, 20)
    ax.tick_params(axis='both', which='major', length=1, width=0.25, labelsize=5, pad=1)

    # Column 3: LFC
    ax = axes[i, 2]
    plot_styled_errorbar(ax, group_fus_tet_on_lfc['motif_num'], group_fus_tet_on_lfc['mean'], group_fus_tet_on_lfc['std'], '-x', 'red', 'FUS-rtTA' if i == 0 else "")
    plot_styled_errorbar(ax, group_tet_on_lfc['motif_num'], group_tet_on_lfc['mean'], group_tet_on_lfc['std'], '-o', 'blue', 'rtTA' if i == 0 else "")
    ax.axhline(y=FUS_WT_LFC, color='red', linestyle='--', label='Standard TRE (with FUS-rtTA)', linewidth=0.5, alpha=0.6)
    ax.axhline(y=TETR_WT_LFC, color='blue', linestyle='--', label='Standard TRE (with rtTA)', linewidth=0.5, alpha=0.6)
    ax.set_ylabel('Mean LFC', fontsize=5)
    ax.set_xlim(1, 8)
    ax.set_ylim(0, 15)
    ax.tick_params(axis='both', which='major', length=1, width=0.25, labelsize=5, pad=1)

    if i == len(GAP_LIST) - 1:
        for j in range(3):
            axes[i, j].set_xlabel('Operator Number', fontsize=5, labelpad=1)
    else:
        for j in range(3):
            axes[i, j].set_xlabel('')

# Global spine width adjustments
for row in axes:
    for ax in row:
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)

# Explicitly grab handles and labels from the first subplot to build the legend cleanly
handles, labels = axes[0, 1].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center', bbox_to_anchor=(0.5, 0.02),
           ncol=2, frameon=False, fontsize=5)

# Column titles at the top
axes[0, 0].set_title('Induced Activity', fontsize=6, pad=3)
axes[0, 1].set_title('Leakiness', fontsize=6, pad=3)
axes[0, 2].set_title('Log fold change', fontsize=6, pad=3)

fig.subplots_adjust(hspace=0.25, wspace=0.3, bottom=0.15, top=0.92, left=0.12, right=0.98)

# --- SAVE (png, svg and pdf, as required) -------------------------------------
# the reference wrote the pdf transparent and the svg opaque; both are kept, and the
# png follows the svg. No bbox_inches: the reference's own margins are the layout.
plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM}.pdf'), format='pdf', transparent=True)
plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM}.svg'), format='svg', transparent=False)
plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM}.png'), format='png', dpi=DPI,
            transparent=False)
plt.close()


# --- REPORT ------------------------------------------------------------------
print(f"[report] both arms merged on (names, seqs): {len(data)} rows")
for arm, cali in (('FUS-rtTA', cali_fus), ('rtTA', cali_tetr)):
    print(f"[report]   {arm:<8s} calibration: induced {cali['induced_mpra_means']:.4f} | "
          f"uninduced {cali['uninduced_mpra_means']:.4f} | fold change {cali['lfc']:.4f}")

for gap_num in GAP_LIST:
    rows = data[data['names'].str.contains(rf'(?=.*{gap_num}gap)(?=.*tetO)', regex=True)]
    counts = rows['names'].str.split('_').str[1].str.extract(r'(\d+)')[0].astype(int)
    series = ', '.join(f'{n}tetO {c}' for n, c in
                       counts.value_counts().sort_index().items())
    print(f"[report] distance {gap_num} bp: {len(rows)} designs ({series})")

print("[report] standard TRE lines, hard-coded against the table:")
for arm, suffix in (('FUS-rtTA', 'FUS'), ('rtTA', 'tetR')):
    rows = data[data['names'].str.contains('positive_wildtype_clontech')]
    print(f"[report]   {arm:<8s} table gives induced "
          f"{rows[f'induced_mpra_means_{suffix}'].mean():.10f} | uninduced "
          f"{rows[f'uninduced_mpra_means_{suffix}'].mean():.10f} | fold "
          f"{rows[f'lfc_{suffix}'].mean():.10f} (n = {len(rows)})")
