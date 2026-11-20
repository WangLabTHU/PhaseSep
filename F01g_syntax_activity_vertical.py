import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# --- NATURE PRESS GLOBAL FONT CONFIGURATION (as in the reference) -------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']  # Preferred journal fonts
plt.rcParams.update({
    'font.size': 5,          # Standard micro-label size
    'axes.labelsize': 5,     # Axis titles
    'axes.titlesize': 5,     # Panel titles
    'xtick.labelsize': 5,    # X tick text
    'ytick.labelsize': 5,    # Y tick text
    'lines.linewidth': 0.25  # Cleaner line weight for small figures
})

# Project layout: data in ./data, figures in ./figs.
DATA_DIR = './data'
FIG_DIR = './figs'
DPI = 300
FIG_STEM = 'F01g_syntax_activity_vertical'

# The four round-1 mean tables under the renamed project file names: the FUS-rtTA arm
# (7_exp_mean_induced / _uninduced) and the rtTA arm (7_exp_mean_E_induced / _uninduced).
FILE_FUS_INDUCED = '7_exp_mean_FUS_TO_induced.tsv'
FILE_FUS_UNINDUCED = '7_exp_mean_FUS_TO_uninduced.tsv'
FILE_TO_INDUCED = '7_exp_mean_TO_induced.tsv'
FILE_TO_UNINDUCED = '7_exp_mean_TO_uninduced.tsv'

# The panel is the six-operator series only; the ligand-bound and apo curves are drawn
# for each gap length, over both flanking designs (deepseed and random).
MOTIF_NUM = 6

# The reference labels the two curves FUS-TetR and TetR. TetR is the repressor; the
# phase-separating transactivator is FUS-rtTA, and the canonical arm is rtTA.
RENAME_LEGEND = True
LEGEND_FUS = 'FUS-rtTA' if RENAME_LEGEND else 'FUS-TetR'
LEGEND_TETR = 'rtTA' if RENAME_LEGEND else 'TetR'


def read_mean_table(filename, value_column, label):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{filename} is missing from {DATA_DIR} (the {label} mean table)")
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = ['seqs', value_column, 'names']
    return df[['seqs', 'names', value_column]]


def calibrate_arm(df, label):
    '''Divide the two mean columns and the fold change by the negative-control mean of
    the same column, exactly as refs_dqx/round1_analyze_degseq.py does. The fold-change
    calibration is the mean of the per-sequence ratios, which is not the ratio of the
    two mean calibrations; both are left as they were.'''
    columns = ['induced_mpra_means', 'uninduced_mpra_means', 'lfc']
    cali = df.loc[df['names'].str.contains('negative_random'), columns].mean()

    # the original divided by these without checking: one missing negative_random row,
    # or a calibration of zero, turns every calibrated value into NaN
    for column in columns:
        if not np.isfinite(cali[column]) or cali[column] <= 0:
            raise ValueError(
                f"the negative_random rows of the {label} arm give a calibration of "
                f"{cali[column]} for {column}, so the whole arm would come out as NaN; "
                f"the original divided by it without checking")

    df = df.copy()
    for column in columns:
        df[column] = df[column] / cali[column]
    return df, cali


def build_arm(label, induced_file, uninduced_file):
    induced = read_mean_table(induced_file, 'induced_mpra_means', label)
    uninduced = read_mean_table(uninduced_file, 'uninduced_mpra_means', label)
    merged = pd.merge(induced, uninduced, on=['names', 'seqs'], how='inner')
    merged = merged[['seqs', 'names', 'induced_mpra_means', 'uninduced_mpra_means']]
    merged['lfc'] = merged['induced_mpra_means'] / merged['uninduced_mpra_means']
    print(f"[report] {label}: {len(induced)} induced + {len(uninduced)} uninduced rows -> "
          f"{len(merged)} merged ({induced_file}, {uninduced_file})")

    merged, cali = calibrate_arm(merged, label)
    print(f"[report]   {label} negative-control calibration: "
          f"induced {cali['induced_mpra_means']:.4f} | "
          f"uninduced {cali['uninduced_mpra_means']:.4f} | "
          f"fold change {cali['lfc']:.4f}")
    return merged, cali


def report_group(tag, frame, arm):
    '''Print the mean induced and uninduced activity, their ratio, and the mean and
    median per-sequence fold of one group, in the calibrated units the figure draws.'''
    induced = frame[f'induced_mpra_means_{arm}']
    uninduced = frame[f'uninduced_mpra_means_{arm}']
    finite = np.isfinite(induced) & np.isfinite(uninduced) & (uninduced > 0)
    induced = induced[finite]
    uninduced = uninduced[finite]
    if induced.empty:
        print(f"[report]   {tag:<28s} no finite rows")
        return
    fold = induced / uninduced
    print(f"[report]   {tag:<28s} n={len(induced):>5d}  "
          f"mean induced {induced.mean():6.2f}  mean uninduced {uninduced.mean():5.2f}  "
          f"ratio {induced.mean() / uninduced.mean():5.2f}  "
          f"mean per-sequence fold {fold.mean():5.2f}  "
          f"median per-sequence fold {np.median(fold):5.2f}")


# --- data ---------------------------------------------------------------------

os.makedirs(FIG_DIR, exist_ok=True)

df_FUS_Trans_merged, cali_FUS = build_arm(
    'FUS-rtTA', FILE_FUS_INDUCED, FILE_FUS_UNINDUCED)
df_tetR_Trans_merged, cali_tetR = build_arm(
    'rtTA', FILE_TO_INDUCED, FILE_TO_UNINDUCED)

# this is the table refs_dqx/round1_analyze_degseq.py writes as degseq_induced.csv
# (apart from the A/M/Z_score/p_value columns, which this figure does not use)
data = pd.merge(df_FUS_Trans_merged, df_tetR_Trans_merged, on=["seqs", "names"],
                suffixes=("_FUS", "_tetR"))
print(f"[report] both arms merged on (names, seqs): {len(data)} sequences, "
      f"{data['names'].nunique()} unique names")


# --- DATA PROCESSING (as in the reference) ------------------------------------

pattern = rf'(?=.*{MOTIF_NUM}tetO)(?=.*gap)'
gap_data = data[data['names'].str.contains(pattern, regex=True)].copy()
if gap_data.empty:
    raise ValueError(
        f"no name in the merged table contains both '{MOTIF_NUM}tetO' and 'gap'; "
        f"the table has {len(data)} rows and the reference figure would have been empty")

label = gap_data['names'].str.split('_', expand=True)
if label.shape[1] != 5:
    raise ValueError(
        f"the {MOTIF_NUM}tetO gap names do not split into the five fields "
        f"'group_motif_gap_flanking_replicate' (got {label.shape[1]} fields); the "
        f"reference assigned five column names unconditionally")
label.columns = ['group', 'motif number', 'gap', 'flanking', 'replicate']

# the reference pattern is not anchored, so '16tetO' would also satisfy '. *6tetO'.
# The library only runs to 11 operators, but fail loudly rather than silently mix a
# 16-operator series into the six-operator panel.
if not (label['motif number'] == f'{MOTIF_NUM}tetO').all():
    bad = label.loc[label['motif number'] != f'{MOTIF_NUM}tetO', 'motif number'].unique()
    raise ValueError(
        f"the unanchored '{MOTIF_NUM}tetO' pattern also selected {list(bad)}; "
        f"anchor the pattern before drawing this panel")

gap_label_data = pd.concat([gap_data, label], axis=1)
gap_label_data['gap_num'] = gap_label_data['gap'].str.extract(r'(\d+)').astype(int)

group_tet_on_expression = gap_label_data.groupby('gap_num')['induced_mpra_means_tetR'].agg(['mean', 'std']).reset_index()
group_fus_tet_on_expression = gap_label_data.groupby('gap_num')['induced_mpra_means_FUS'].agg(['mean', 'std']).reset_index()
group_tet_on_leakiness = gap_label_data.groupby('gap_num')['uninduced_mpra_means_tetR'].agg(['mean', 'std']).reset_index()
group_fus_tet_on_leakiness = gap_label_data.groupby('gap_num')['uninduced_mpra_means_FUS'].agg(['mean', 'std']).reset_index()

# --- EXACT CANVAS DIMENSIONS (Vertical Stack) ---------------------------------
width_inch = 65 / 25.4
height_inch = 59 / 25.4
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width_inch, height_inch), sharex=True)

# --- TOP PANEL: Induction ------------------------------------------------------

# FUS-rtTA (cross markers)
ax1.plot(group_fus_tet_on_expression['gap_num'], group_fus_tet_on_expression['mean'],
         '-x', color='#012f48', alpha=0.8, markersize=3, label=LEGEND_FUS)
# rtTA (circle markers)
ax1.plot(group_tet_on_expression['gap_num'], group_tet_on_expression['mean'],
         ':o', color='#012f48', alpha=0.8, markersize=2.5, markerfacecolor='none',
         markeredgecolor='#012f48', markeredgewidth=0.5, label=LEGEND_TETR)

ax1.tick_params(axis='y', labelcolor='#012f48', pad=1)
ax1.set_title('Induced Activity', color='#012f48', fontweight='bold', pad=3)
ax1.set_xlim(0, 28)
ax1.set_ylim(0, 50)

# --- BOTTOM PANEL: Leakiness ---------------------------------------------------

# FUS-rtTA (cross markers)
ax2.plot(group_fus_tet_on_leakiness['gap_num'], group_fus_tet_on_leakiness['mean'],
         '-x', color='#669aba', alpha=0.8, markersize=3, label=LEGEND_FUS)
# rtTA (circle markers)
ax2.plot(group_tet_on_leakiness['gap_num'], group_tet_on_leakiness['mean'],
         ':o', color='#669aba', alpha=0.8, markersize=2.5, markerfacecolor='none',
         markeredgecolor='#669aba', markeredgewidth=0.5, label=LEGEND_TETR)

ax2.tick_params(axis='y', labelcolor='#669aba', pad=0.5)
ax2.tick_params(axis='x', pad=1)
ax2.set_title('Leakiness', color='#669aba', fontweight='bold', pad=3)
ax2.set_xlabel('Inter-operator distance', labelpad=0.8)  # Only bottom panel has x-label
ax2.set_xlim(0, 28)
ax2.set_ylim(0, 20)

# --- SINGLE SHARED Y-AXIS LABEL ---
fig.supylabel('Mean calibrated MPRA activity', fontweight='bold', fontsize=5, x=0.08)

# --- GLOBAL TICK, SPINE & LAYOUT CONFIGURATION ---------------------------------

for ax in [ax1, ax2]:
    ax.tick_params(axis='both', which='major', length=1, width=0.25)

    for spine in ax.spines.values():
        spine.set_linewidth(0.25)

    ax.legend(frameon=False, fontsize=4, handletextpad=0.2, loc='upper right')

# Remove x-axis ticks from the top panel
ax1.tick_params(axis='x', which='both', bottom=False, top=False)

fig.subplots_adjust(hspace=0.22, bottom=0.18, top=0.95, left=0.18, right=0.95)

# --- SAVE (png, svg and pdf, as required) --------------------------------------

plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM}.png'), format='png', dpi=DPI,
            transparent=True)
plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM}.svg'), format='svg', transparent=True)
plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM}.pdf'), format='pdf', transparent=True)
plt.close()


# --- REPORT --------------------------------------------------------------------
# Diagnostics only: nothing below changes the figure. The calibration constants are the
# raw negative-control means of the two arms, so the raw fold of any sequence is
#   calibrated fold x (cali_induced / cali_uninduced)
# and the reader can see whether the negative-control calibration is what makes the
# induction look small.

print()
print("[report] calibration and what it does to the fold")
for arm_label, cali in (('FUS-rtTA', cali_FUS), ('rtTA', cali_tetR)):
    factor = cali['induced_mpra_means'] / cali['uninduced_mpra_means']
    print(f"[report]   {arm_label:<8s} cali(induced) {cali['induced_mpra_means']:.4f}  "
          f"cali(uninduced) {cali['uninduced_mpra_means']:.4f}  "
          f"cali(uninduced)/cali(induced) {1 / factor:.4f}  "
          f"raw fold = calibrated fold x {factor:.4f}")

print()
print(f"[report] the {MOTIF_NUM}tetO gap panel (the rows this figure draws)")
report_group('FUS-rtTA, both conditions', gap_label_data, 'FUS')
report_group('rtTA, both conditions', gap_label_data, 'tetR')
report_group('whole spacer library, FUS', data[data['names'].str.startswith('spacer_')], 'FUS')
report_group('whole spacer library, rtTA', data[data['names'].str.startswith('spacer_')], 'tetR')

# the internal positive control is the best yardstick for "how many fold can this assay
# report at all"; the Clontech rows are one sequence repeated, so n is not n replicates
print()
print("[report] internal controls (calibrated, FUS-rtTA arm)")
for tag in ('positive_wildtype_clontech', 'positive_wildtype_pwx158',
            'positive_wildtype_random', 'negative_random', 'negative_hek293'):
    rows = data[data['names'].str.contains(tag)]
    if rows.empty:
        print(f"[report]   {tag:<28s} absent")
        continue
    report_group(tag, rows, 'FUS')
    print(f"[report]     {tag:<26s} rows {len(rows)}, unique sequences {rows['seqs'].nunique()}")

print()
print(f"[report] the uninduced mean is measured against a negative control of 1.00, so "
      f"the floor is 1 unit, i.e. 1 / uninduced of the uninduced mean; subtracting it "
      f"changes the FUS fold from "
      f"{gap_label_data['induced_mpra_means_FUS'].mean() / gap_label_data['uninduced_mpra_means_FUS'].mean():.2f} "
      f"to "
      f"{(gap_label_data['induced_mpra_means_FUS'].mean() - 1) / (gap_label_data['uninduced_mpra_means_FUS'].mean() - 1):.2f}")
