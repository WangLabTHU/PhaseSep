import os
import re
from functools import reduce

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'

DATA_DIR = './data'
FIG_DIR = './figs'

# The original script inner-joins the two round-1 mean tables on (names, seqs),
# which it called 7_exp_mean_induced.tsv / 7_exp_mean_uninduced.tsv. On a root
# that only carries the per-replicate tables, point these at
# 7_exp_rep1_FUS_TO_induced.tsv / 7_exp_rep1_FUS_TO_uninduced.tsv instead.
FILE_INDUCED = '7_exp_mean_FUS_TO_induced.tsv'
FILE_UNINDUCED = '7_exp_mean_FUS_TO_uninduced.tsv'

# The designed library written by R01_round1_prepare_lib_combine.py: 18,000 designs,
# 11,850 of them spacers, 50 designs in every designed (tetO, gap) cell. The spacer
# panel plots this design rather than the sequences that survived sequencing, which
# hold only 11,425 of the 11,850.
DESIGN_CSV = './results/round1_library/final_library.csv'

FIG_STEM_CONSTRUCTION = 'S01_library_construction'
FIG_STEM_SPACER = 'S01_spacer_template'
DPI = 600

# Number of nucleotides of one tetO motif, measured on the spacer sequences
# ('TCCCTATCAGTGATAGAGA'); used only for the motif-array length reported at the end.
TETO_MOTIF_LEN = 19

SPACER_CMAP = sns.light_palette('#74a892', as_cmap=True)
SPACER_MARGINAL_COLOR = '#74a892'
NOT_DESIGNED_COLOR = '#e6e6e6'

os.makedirs(FIG_DIR, exist_ok=True)


def assign_group(name):
    name = str(name).lower()
    if "spacer_" in name:
        return "spacer_template"
    elif "mutant_" in name:
        if re.search(r"mutant_\w+_\d+\w+_\d+\w+", name):
            return "mutant_double"
        elif re.search(r"mutant_\w+_\d+\w+", name):
            return "mutant_single"
    elif "flanking_deepseed" in name:
        return "flanking_deepseed"
    elif "flanking_random" in name:
        return "flanking_random"
    elif "positive_wildtype_pwx158_" in name:
        return "positive_wildtype_pwx158_ntetO"
    elif "positive_wildtype_random_" in name:
        return "positive_wildtype_random_ntetO"
    elif "positive_wildtype_clontech_" in name:
        return "positive_wildtype_clontech_7tetO"
    elif "positive_deepseed_comb" in name:
        return "positive_deepseed_comb"
    elif "negative_hek293_enhancer" in name:
        return "negative_hek293_enhancer"  # from training data
    elif "negative_hek293_nomotifs" in name:
        return "negative_hek293_nomotifs"
    elif "negative_random" in name:
        return "negative_random"
    else:
        return "other"


def extract_features(name):
    name = str(name).lower()
    teto_match = re.search(r'(\d+)teto', name)
    gap_match = re.search(r'(\d+)gap', name)
    teto_num = int(teto_match.group(1)) if teto_match else None
    gap_num = int(gap_match.group(1)) if gap_match else None
    return teto_num, gap_num


GROUP_ORDER = [
    "spacer_template",
    "mutant_single",
    "mutant_double",
    "flanking_deepseed",
    "flanking_random",
    "positive_wildtype_pwx158_ntetO",
    "positive_wildtype_random_ntetO",
    "positive_wildtype_clontech_7tetO",
    "positive_deepseed_comb",
    "negative_hek293_enhancer",
    "negative_hek293_nomotifs",
    "negative_random",
]

# --- data ---------------------------------------------------------------------

df_induced = pd.read_csv(os.path.join(DATA_DIR, FILE_INDUCED), sep="\t", header=None)
df_induced.columns = ["seqs", "induced_mpra_means", "names"]
df_uninduced = pd.read_csv(os.path.join(DATA_DIR, FILE_UNINDUCED), sep="\t", header=None)
df_uninduced.columns = ["seqs", "uninduced_mpra_means", "names"]

df_merged = reduce(lambda left, right: pd.merge(left, right, on=['names', 'seqs'], how='inner'),
                   [df_induced, df_uninduced])
df_merged["group"] = df_merged["names"].apply(assign_group)


# --- Fig. 2b (left): library composition --------------------------------------

group_counts = df_merged['group'].value_counts().reset_index()
group_counts.columns = ['group', 'count']
group_counts['group'] = pd.Categorical(group_counts['group'], categories=GROUP_ORDER, ordered=True)
group_counts = group_counts.sort_values('group').dropna()


# --- Fig. 2b (right): spacer template design space ----------------------------
#
# Every spacer construct is 252 nt: n copies of the tetO motif separated by n-1
# gaps of `gap` nt, the gaps filled either with a designed (deepseed) or a random
# sequence. The design tiles a triangular region of the (n, gap) plane because
# the motif array has to fit into the construct; a cell holds exactly 50 designs,
# so the cells are now drawn in one colour that only marks whether the (n, gap)
# combination was designed, and the marginals carry the counts per n and per gap.

if not os.path.exists(DESIGN_CSV):
    raise FileNotFoundError(
        f"{DESIGN_CSV} not found; it is written by R01_round1_prepare_lib_combine.py. "
        "Copy final_library.csv from that run if the design step was executed elsewhere.")
design = pd.read_csv(DESIGN_CSV).rename(columns={'tags': 'names'})
df_spacer = design[design['names'].map(assign_group) == 'spacer_template'].copy()
df_spacer['tetO_num'], df_spacer['gap_num'] = zip(*df_spacer['names'].apply(extract_features))
df_spacer = df_spacer.dropna(subset=['tetO_num', 'gap_num'])
df_spacer['fill'] = df_spacer['names'].str.lower().str.extract(r'gap_([a-z]+)_')[0]

tetO_values = np.arange(int(df_spacer['tetO_num'].min()), int(df_spacer['tetO_num'].max()) + 1)
gap_values = np.arange(int(df_spacer['gap_num'].min()), int(df_spacer['gap_num'].max()) + 1)

grid = pd.crosstab(df_spacer['gap_num'], df_spacer['tetO_num'])
grid = grid.reindex(index=gap_values, columns=tetO_values, fill_value=0)
n_gap, n_tetO = len(gap_values), len(tetO_values)

# axes laid out in inch coordinates so the design cells are square
cell = 0.11
pad_left, pad_bottom, pad_top, pad_right = 0.58, 0.62, 0.18, 0.14
marginal_top_h, marginal_right_w = 0.85, 0.95
# the marginals are kept slightly apart from the grid, otherwise the outermost
# tick labels of the two axes meet at the shared corner
gap_top, gap_right = 0.07, 0.06

grid_w = n_gap * cell
grid_h = n_tetO * cell
width = pad_left + grid_w + gap_right + marginal_right_w + pad_right
height = pad_bottom + grid_h + gap_top + marginal_top_h + pad_top
fig = plt.figure(figsize=(width, height))

ax_main = fig.add_axes([pad_left / width, pad_bottom / height, grid_w / width, grid_h / height])
ax_top = fig.add_axes([pad_left / width, (pad_bottom + grid_h + gap_top) / height,
                       grid_w / width, marginal_top_h / height], sharex=ax_main)
ax_right = fig.add_axes([(pad_left + grid_w + gap_right) / width, pad_bottom / height,
                         marginal_right_w / width, grid_h / height], sharey=ax_main)

# binary encoding: a designed cell takes the green the count scale used at its top
# end, an un-designed cell keeps the grey drawn underneath, so the colours of the
# finished figure are unchanged
designed = (grid > 0)
sns.heatmap(designed.T.astype(float), mask=~designed.T.values,
            cmap=ListedColormap([SPACER_CMAP(1.0)]), vmin=0, vmax=1,
            linewidths=0.25, linecolor='white', cbar=False, ax=ax_main,
            xticklabels=[str(v) for v in gap_values],
            yticklabels=[str(v) for v in tetO_values])
# the empty cells of the (tetO, gap) plane: the motif array would not fit into the
# construct, so no sequence was designed there. Drawn as an artist because
# transparent=True would drop the axes background.
ax_main.add_patch(Rectangle((-0.5, -0.5), n_gap, n_tetO,
                            facecolor=NOT_DESIGNED_COLOR, edgecolor='none', zorder=0))
# the masked cells keep their edges, which would show as a white lattice over the
# grey; cover each run of un-designed cells again above the mesh
for row, values in enumerate(grid.values.T):
    run_start = None
    for col, value in enumerate(list(values) + [1]):
        if value == 0 and run_start is None:
            run_start = col
        elif value != 0 and run_start is not None:
            ax_main.add_patch(Rectangle((run_start, row), col - run_start, 1,
                                        facecolor=NOT_DESIGNED_COLOR, edgecolor='none', zorder=3))
            run_start = None
ax_main.invert_yaxis()  # tetO copies increase upwards, the triangle opens to the right
ax_main.set_xlabel('Inter-operator distances (bp)', fontsize=6, labelpad=2)
ax_main.set_ylabel('Operator number', fontsize=6, labelpad=2)
ax_main.tick_params(axis='both', labelsize=5, length=1, width=0.25)
plt.setp(ax_main.get_xticklabels(), rotation=0)
# 48 gap columns: label every other round value instead of all of them
kept = [i for i, value in enumerate(gap_values) if value in [3, 5] + list(range(10, 51, 5))]
ax_main.set_xticks([i + 0.5 for i in kept])
ax_main.set_xticklabels([str(gap_values[i]) for i in kept])
for spine in ax_main.spines.values():
    spine.set_linewidth(0.25)
sns.despine(ax=ax_main)

# the marginals sit on the halves of the unit cells, as the heatmap cells do
x_pos = np.arange(n_gap) + 0.5
y_pos = np.arange(n_tetO) + 0.5
gap_totals = grid.values.sum(axis=1)
tetO_totals = grid.values.sum(axis=0)

ax_top.bar(x_pos, gap_totals, width=0.72, color=SPACER_MARGINAL_COLOR, linewidth=0)
ax_top.set_ylabel('Sequences', fontsize=5, labelpad=1)
ax_top.tick_params(axis='y', labelsize=5, length=1, width=0.25)
ax_top.tick_params(axis='x', labelbottom=False, length=0)
sns.despine(ax=ax_top)

ax_right.barh(y_pos, tetO_totals, height=0.72, color=SPACER_MARGINAL_COLOR, linewidth=0)
# all counts in one column right of the longest bar rather than one per bar end
label_x = tetO_totals.max() * 1.04
for position, total in zip(y_pos, tetO_totals):
    ax_right.text(label_x, position, f'{int(total)}', ha='left', va='center', fontsize=4.5)
ax_right.set_xlim(0, tetO_totals.max() * 1.38)
ax_right.tick_params(axis='both', length=0, labelbottom=False, labelleft=False)
sns.despine(ax=ax_right, left=True, bottom=True)

plt.savefig(os.path.join(FIG_DIR, FIG_STEM_SPACER + '.pdf'), dpi=DPI, bbox_inches='tight', transparent=True)
plt.savefig(os.path.join(FIG_DIR, FIG_STEM_SPACER + '.svg'), dpi=DPI, bbox_inches='tight', transparent=True)
plt.savefig(os.path.join(FIG_DIR, FIG_STEM_SPACER + '.png'), dpi=DPI, bbox_inches='tight', transparent=True)
plt.close()


# --- REPORT ------------------------------------------------------------------

print("[report] Library composition (sequences in both round-1 mean tables):")
total = int(group_counts['count'].sum())
for name, count in zip(group_counts['group'], group_counts['count']):
    print(f"[report]   {name}: {int(count)} ({100 * count / total:.1f}%)")
print(f"[report]   total: {total}")

per_cell = grid.values[grid.values > 0]
print("[report] Spacer template:")
print(f"[report]   sequences {int(len(df_spacer))}, design cells {int(per_cell.size)} "
      f"(tetO {tetO_values[0]}-{tetO_values[-1]} x gap {gap_values[0]}-{gap_values[-1]})")
print(f"[report]   sequences per cell: min {int(per_cell.min())} / median {np.median(per_cell):.0f} / "
      f"max {int(per_cell.max())}")
print("[report]   gap fill: " + ", ".join(f"{k} {v}" for k, v in df_spacer['fill'].value_counts().items()))

row_idx, col_idx = np.where(grid.values > 0)
n_arr = tetO_values[col_idx]
g_arr = gap_values[row_idx]
array_len = n_arr * TETO_MOTIF_LEN + (n_arr - 1) * g_arr
print(f"[report]   longest motif array {int(array_len.max())} nt "
      f"(tetO {int(n_arr[array_len.argmax()])}, gap {int(g_arr[array_len.argmax()])}) "
      f"inside the 252 nt construct")
