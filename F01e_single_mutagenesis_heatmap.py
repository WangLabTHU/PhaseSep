import os
from functools import reduce

import joblib
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Project layout: data in ./data, figures in ./figs.
DATA_DIR = './data'
FIG_DIR = './figs'
FIG_STEM = 'F01e_single_mutagenesis_heatmap'
DPI = 600

REP_FILES = {
    'induced_rep1': '7_exp_rep1_FUS_TO_induced.tsv',
    'induced_rep2': '7_exp_rep2_FUS_TO_induced.tsv',
    'uninduced_rep1': '7_exp_rep1_FUS_TO_uninduced.tsv',
    'uninduced_rep2': '7_exp_rep2_FUS_TO_uninduced.tsv',
}
WT_NAME = 'positive_wildtype_clontech_7tetO_rep0'
WT_SEQUENCE = 'tccctatcagtgatagaga'
EQUIVALENCE_MARGIN = 0.5
TOST_DF = 58

# ==========================================
# --- ADJUSTABLE VISUAL PARAMETERS HERE ---
# ==========================================
FRAME_LINE_WIDTH = 0.25      # 1. Thickness of outer frames/spines
GRID_LINE_WIDTH = 0.25       # 1. Thickness of internal heatmap grid lines
DOT_BORDER_WIDTH = 0         # 1. Thickness of the significance dot borders
SLASH_LINE_WIDTH = 0.5       # Thickness of wild-type diagonal red slashes

LABEL_PAD_1 = 1.1            # 2. Horizontal position of the coordinate numbers (1, 2, 3...)
LABEL_PAD_2 = 2.0            # 2. Horizontal position of the bases (A, T, C, G...)
# ==========================================

# --- NATURE PRESS GLOBAL FONT CONFIGURATION (as in the reference) -------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams.update({
    'font.size': 5,
    'axes.labelsize': 5,
    'axes.titlesize': 5,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'lines.linewidth': FRAME_LINE_WIDTH,
    'axes.linewidth': FRAME_LINE_WIDTH,     # Set global spine thickness
    'patch.linewidth': FRAME_LINE_WIDTH,    # Set global legend box border thickness
    'pdf.fonttype': 'truetype',
    'svg.fonttype': 'none'
})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


# --- STEP 3.0: CALIBRATE THE FOUR ROUND-1 REPLICATE TABLES --------------------

def read_rep_table(filename, column):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{filename} is missing from {DATA_DIR}; this is one of the four round-1 "
            "replicate tables the figure is built from.")
    df = pd.read_csv(path, sep='\t', header=None)
    if df.shape[1] != 3:
        raise ValueError(
            f"{filename} has {df.shape[1]} columns; expected the three columns "
            "'seq, value, name'.")
    df.columns = ['seqs', column, 'names']
    return df[['seqs', 'names', column]]


rep_frames = [read_rep_table(filename, column) for column, filename in REP_FILES.items()]
calibrated = reduce(lambda left, right: pd.merge(left, right, on=['names', 'seqs'],
                                                 how='inner'), rep_frames)
if calibrated.empty:
    raise ValueError(
        "the four replicate tables share no (names, seqs) rows; the reference merged "
        "them with an inner join and would have drawn an empty figure.")

rep_columns = list(REP_FILES)
cali = calibrated.loc[calibrated['names'].str.contains('negative_random'),
                      rep_columns].mean()
for column in rep_columns:
    if not np.isfinite(cali[column]) or cali[column] <= 0:
        raise ValueError(
            f"the negative_random rows give a calibration of {cali[column]} for "
            f"{column}; every calibrated replicate would be NaN or sign-flipped.")

calibrated = calibrated.copy()
for column in rep_columns:
    calibrated[column] = calibrated[column] / cali[column]

# --- STEP 3.0: SINGLE-MUTANT SET AND WT ---------------------------------------

wt_set = calibrated[calibrated['names'] == WT_NAME]
if len(wt_set) != 1:
    raise ValueError(
        f"expected exactly one {WT_NAME!r} row in the calibrated table, found "
        f"{len(wt_set)}; the WT row anchors both the TOST comparison and the red "
        "wild-type slashes.")

mutant_set = calibrated[calibrated['names'].str.contains(r'mutant_O2', regex=True)]
label = mutant_set['names'].str.split('_', expand=True)
if label.shape[1] != 4:
    raise ValueError(
        "the mutant_O2 names do not split into the four fields "
        f"'group_motif_1st_mutant_2nd_mutant' (got {label.shape[1]}); the reference "
        "assigned four column names unconditionally.")
label.columns = ['group', 'motif_name', '1st_mutant', '2nd_mutant']
# the extracted digits arrive as an object column; older pandas cannot cast object
# strings straight to the nullable Int64 dtype, so go through pd.to_numeric first
# (it yields float with NaN where the field is absent, which int() accepts below).
label['1st_position'] = pd.to_numeric(
    label['1st_mutant'].str.extract(r'(\d+)', expand=False), errors='coerce')
label['1st_mutated_base'] = label['1st_mutant'].str.extract(r'([A-Za-z]+)', expand=False)
label['2nd_position'] = pd.to_numeric(
    label['2nd_mutant'].str.extract(r'(\d+)', expand=False), errors='coerce')
label['2nd_mutated_base'] = label['2nd_mutant'].str.extract(r'([A-Za-z]+)', expand=False)

mutant_label = pd.concat([mutant_set, label['1st_position'], label['1st_mutated_base'],
                          label['2nd_position'], label['2nd_mutated_base']], axis=1)

# the reference counted non-null cells over its 9 calibrated columns plus the four
# parsed mutation fields, so a single mutant scored 11; here we count the four
# mutation fields directly (two non-null = single mutation).
non_null_counts = mutant_label[['1st_position', '1st_mutated_base',
                                '2nd_position', '2nd_mutated_base']].notnull().sum(axis=1)
single_mutant = mutant_label[non_null_counts == 2].copy()
single_mutant = single_mutant.iloc[:, :-2]
single_mutant = single_mutant.reindex(columns=['names', 'induced_rep1', 'induced_rep2',
                                               'uninduced_rep1', 'uninduced_rep2',
                                               '1st_position', '1st_mutated_base'])
single_mutant_data = pd.concat([single_mutant, wt_set], ignore_index=True)

if len(single_mutant_data) != TOST_DF:
    raise ValueError(
        f"the single-mutant set plus WT has {len(single_mutant_data)} rows; the "
        f"reference TOST uses df={TOST_DF} (57 mutants + WT), so this table is not "
        "the one the figure was made from.")

# --- STEP 3.1: TOST SIMILARITY STATISTIC (reference logic, unchanged) ---------


def benjamini_hochberg(p_values):
    '''Benjamini-Hochberg adjusted p-values, the same numbers scipy.stats
    .false_discovery_control(..., method="bh") returns; that function only exists in
    scipy >= 1.11, so the formula is applied here instead.'''
    p = np.asarray(p_values, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * (n / np.arange(1, n + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def run_python_limma_similarity_logic(df, column1, column2, margin=0.5):
    '''df: 57 mutants and the WT last; columns hold the two calibrated replicates.'''
    data = df[[column1, column2]].copy()
    log_data = np.log2(data + 1e-6)

    # Variance for each row (n=2)
    row_vars = log_data.var(axis=1, ddof=1)

    # Empirical-Bayes step: global prior = median row variance
    global_variance_prior = np.median(row_vars.dropna())
    moderated_vars = (row_vars + global_variance_prior) / 2

    # Mutants vs WT (WT is the last row)
    wt_mean = log_data.iloc[-1].mean()
    mutant_means = log_data.iloc[:-1].mean(axis=1)
    logFC = mutant_means - wt_mean

    se = np.sqrt(moderated_vars.iloc[:-1] * (1 / 2 + 1 / 2))  # n=2 vs n=2

    # TOST: logFC significantly above -margin and below +margin
    t_lower = (logFC - (-margin)) / se
    p_lower = 1 - stats.t.cdf(t_lower, df=TOST_DF)
    t_upper = (logFC - margin) / se
    p_upper = stats.t.cdf(t_upper, df=TOST_DF)
    p_vals = np.maximum(p_lower, p_upper)

    adj_p_vals = benjamini_hochberg(p_vals)

    return pd.DataFrame({
        'names': df.iloc[:-1]['names'],
        'logFC': logFC,
        'P.Value_Similarity': p_vals,
        'adj.P.Val_Similarity': adj_p_vals,
    })


induced_stats = run_python_limma_similarity_logic(
    single_mutant_data, 'induced_rep1', 'induced_rep2', margin=EQUIVALENCE_MARGIN)
uninduced_stats = run_python_limma_similarity_logic(
    single_mutant_data, 'uninduced_rep1', 'uninduced_rep2', margin=EQUIVALENCE_MARGIN)

df_limma_combined = pd.merge(induced_stats, uninduced_stats, on='names', how='inner',
                             suffixes=('_induced', '_uninduced'))
complete_stats = pd.merge(df_limma_combined, single_mutant_data, on='names', how='inner')
complete_stats = complete_stats[['names', 'logFC_induced', 'adj.P.Val_Similarity_induced',
                                 'logFC_uninduced', 'adj.P.Val_Similarity_uninduced',
                                 '1st_position', '1st_mutated_base']]


# --- STEP 3.2: HEATMAP (reference plotting code) ------------------------------


def plot_mutagenesis_heatmap_vertical(df, wt_sequence):
    base_order = ['A', 'T', 'C', 'G']
    num_positions = len(wt_sequence)
    wt_sequence_upper = wt_sequence.upper()

    # --- 1. Separate 4-Column Mutation Matrices & 1-Column HTH Array ---
    matrix_induced = np.full((num_positions, 4), np.nan)
    matrix_uninduced = np.full((num_positions, 4), np.nan)
    matrix_p_induced = np.full((num_positions, 4), np.nan)
    matrix_p_uninduced = np.full((num_positions, 4), np.nan)

    for _, row in df.iterrows():
        p_idx = int(row['1st_position']) - 1
        if row['1st_mutated_base'] in base_order and 0 <= p_idx < num_positions:
            b_idx = base_order.index(row['1st_mutated_base'])
            matrix_induced[p_idx, b_idx] = row['logFC_induced']
            matrix_uninduced[p_idx, b_idx] = row['logFC_uninduced']
            matrix_p_induced[p_idx, b_idx] = row['adj.P.Val_Similarity_induced']
            matrix_p_uninduced[p_idx, b_idx] = row['adj.P.Val_Similarity_uninduced']

    hth_col = np.full((num_positions, 1), -555.0)
    hth_col[3:8, 0] = -999.0
    hth_col[12:17, 0] = -999.0

    # --- 2. Vertical Page Layout Setup ---
    # the positions run down the page instead of across it, so the three tracks that
    # were stacked as rows are now laid out side by side as columns, the position and
    # base labels sit to the right of the HTH track, and the colour bar runs
    # horizontally below the functional panels; the page is a little wider than the
    # transposed 45 mm so that the two track titles fit over columns that are only
    # four cells wide
    width_inch = 50 / 25.4
    height_inch = 70 / 25.4

    fig = plt.figure(figsize=(width_inch, height_inch), layout='constrained')

    # 1. Master grid: the two rows are the heatmap tracks and the slot the colour bar
    #    fills below them, the two columns are the functional panels and the HTH track
    hspace_cbar = 0.12
    gs_master = gridspec.GridSpec(2, 2, height_ratios=[1, 0.03], width_ratios=[8, 1],
                                  hspace=hspace_cbar, wspace=0.05)

    # 2. Sub-grid for the left two panels (Induction and Leakiness)
    wspace_functional = 0.02
    gs_top = gs_master[0, 0].subgridspec(1, 2, width_ratios=[1, 1],
                                         wspace=wspace_functional)

    ax_ind = fig.add_subplot(gs_top[0, 0])
    ax_leak = fig.add_subplot(gs_top[0, 1], sharey=ax_ind)
    ax_hth = fig.add_subplot(gs_master[0, 1], sharey=ax_ind)

    # --- 3. Custom Color Map Profile ---
    positions = [0.0, 2.0 / 6.0, 4.0 / 6.0, 1.0]
    gradient_colors = ["#080637", "#236BAA", "white", "#933030"]
    pbr_cmap = mcolors.LinearSegmentedColormap.from_list(
        "TBBR", list(zip(positions, gradient_colors)))
    pbr_cmap.set_bad(color='white')
    shared_norm = plt.Normalize(vmin=-4.0, vmax=2.0)

    # --- 4. Render Functional Heatmaps (Induction & Leakiness) ---
    panels_config = [
        (ax_ind, matrix_induced, "Induced Activity", "#012f48", matrix_p_induced),
        (ax_leak, matrix_uninduced, "Leakiness", "#669aba", matrix_p_uninduced),
    ]

    for ax, matrix, title, color_str, p_matrix in panels_config:
        im = ax.imshow(matrix, cmap=pbr_cmap, norm=shared_norm, aspect='equal',
                       origin='upper', zorder=1)

        ax.xaxis.tick_top()
        x_obj = ax.set_title(title, fontweight='bold', fontsize=5.5, pad=3)
        x_obj.set_color(color_str)

        ax.set_xticks(range(4))
        ax.set_xticklabels(base_order, fontweight='bold')
        ax.tick_params(axis='x', which='major', length=1.5, width=FRAME_LINE_WIDTH, pad=1)

        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(num_positions - 0.5, -0.5)

        # Grid boundaries
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_positions, 1), minor=True)
        ax.grid(which='minor', color='#555555', linestyle='-', linewidth=GRID_LINE_WIDTH)
        ax.tick_params(axis='both', which='minor', length=0, width=0)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.legend(frameon=False)

        # Draw Red Diagonal Slashes for Wild-Type (non-mutated) positions
        for p_idx, wt_base in enumerate(wt_sequence_upper):
            if wt_base in base_order:
                x = base_order.index(wt_base)
                ax.plot([x - 0.5, x + 0.5], [p_idx - 0.5, p_idx + 0.5], color='red',
                        linewidth=SLASH_LINE_WIDTH, zorder=3)

        # Significance Dots
        for y in range(num_positions):
            for x in range(4):
                pval = p_matrix[y, x]
                if pd.isna(pval):
                    continue
                if pval < 0.001:
                    ax.scatter([x, x], [y - 0.14, y + 0.14], s=0.8, c='black',
                               edgecolors='black', linewidths=DOT_BORDER_WIDTH, zorder=4)
                elif pval < 0.01:
                    ax.scatter([x], [y], s=0.8, c='black', edgecolors='black',
                               linewidths=DOT_BORDER_WIDTH, zorder=4)

    # --- 5. Render Independent HTH Track Heatmap ---
    hth_bind_mask = np.ma.masked_where(hth_col != -999, hth_col)
    hth_free_mask = np.ma.masked_where(hth_col != -555, hth_col)

    ax_hth.imshow(hth_bind_mask, cmap=mcolors.ListedColormap(['#808080']),
                  aspect='equal', origin='upper', zorder=2)
    ax_hth.imshow(hth_free_mask, cmap=mcolors.ListedColormap(['#ffffff']),
                  aspect='equal', origin='upper', zorder=2)

    ax_hth.xaxis.tick_top()
    ax_hth.set_xticks([0])
    ax_hth.set_xticklabels(['HTH'], fontweight='bold')
    ax_hth.tick_params(axis='x', which='major', length=1.5, width=FRAME_LINE_WIDTH, pad=1)
    ax_hth.set_xlim(-0.5, 0.5)

    ax_hth.set_xticks(np.arange(-0.5, 1, 1), minor=True)
    ax_hth.set_yticks(np.arange(-0.5, num_positions, 1), minor=True)
    ax_hth.grid(which='minor', color='#555555', linestyle='-', linewidth=GRID_LINE_WIDTH)
    ax_hth.tick_params(axis='both', which='minor', length=0, width=0)
    ax_hth.tick_params(axis='y', which='both', left=False, labelleft=False)

    # --- 6. Set Uniform Shared Y-Axis Ticks on the Right (HTH Ax) ---
    ax_hth.set_yticks(range(num_positions))
    ax_hth.set_yticklabels([])
    ax_hth.tick_params(axis='y', which='major', right=True, length=1.5,
                       width=FRAME_LINE_WIDTH)

    # Custom sequence labels tracking utilizing padding configuration variables
    for idx, base in enumerate(wt_sequence):
        ax_hth.text(LABEL_PAD_1, idx, str(idx + 1), ha='center', va='center',
                    fontsize=4.5, fontname='Arial', clip_on=False)
        ax_hth.text(LABEL_PAD_2, idx, base.upper(), ha='center', va='center',
                    fontsize=5.5, fontweight='bold', fontname='Arial', clip_on=False)

    # --- 7. Unframed Legend text notification below the HTH track ---
    ax_hth.text(0, num_positions - 0.2, "■ HTH binding", ha='center', va='top',
                color='#808080', fontsize=4.5, fontweight='bold', clip_on=False)

    # --- 8. Horizontal Colorbar Layout, Below the Functional Panels ---
    # the bar keeps the grid slot it had (same length and distance below the panels)
    # but is centred on the drawn figure: the panels sit left of centre because the
    # position and base labels stand to the right of the HTH track, so the slot alone
    # would leave the bar off centre
    fig.canvas.draw()
    slot = gs_master[1, 0].get_position(fig)
    content = fig.get_tightbbox(fig.canvas.get_renderer())
    centre = (content.x0 + content.x1) / 2 / fig.get_figwidth()
    cbar_ax = fig.add_axes([centre - slot.width / 2, slot.y0,
                            slot.width, slot.height])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=5, length=1.5, width=FRAME_LINE_WIDTH, pad=1)
    cbar.outline.set_linewidth(FRAME_LINE_WIDTH)
    cbar.set_label(r'Relative calibrated MPRA activity', fontweight='bold',
                   labelpad=4, fontsize=5)

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM}.pdf'), format='pdf',
                bbox_inches='tight', transparent=False)
    plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM}.svg'), format='svg',
                bbox_inches='tight', transparent=False)
    plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM}.png'), format='png',
                bbox_inches='tight', transparent=False, dpi=DPI)
    plt.close()


plot_mutagenesis_heatmap_vertical(complete_stats, WT_SEQUENCE)

print(f"[report] negative-control calibration (per replicate): "
      + ", ".join(f"{column} {cali[column]:.4f}" for column in rep_columns))
print(f"[report] single-mutant set: {len(single_mutant)} mutants + 1 WT = "
      f"{len(single_mutant_data)} rows; TOST margin {EQUIVALENCE_MARGIN} log2, df={TOST_DF}")
print(f"[report] identical-to-WT calls (adj.P < 0.05): "
      f"induced {(complete_stats['adj.P.Val_Similarity_induced'] < 0.05).sum()}, "
      f"uninduced {(complete_stats['adj.P.Val_Similarity_uninduced'] < 0.05).sum()} "
      f"of {len(complete_stats)}")
