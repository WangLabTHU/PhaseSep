import os
from functools import reduce

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'

DATA_DIR = './data'
FIG_DIR = './figs'
DPI = 300

# The offset every log2 in this script takes: a level is log2(value + LOG_OFFSET), a fold
# change is log2((y + LOG_OFFSET) / (x + LOG_OFFSET)). It is 1e-6, the same offset the limma
# statistic of the original has always used, so the barplot, the combination scatter and the
# volcano share one convention. LOG_OFFSET_LABEL is what the axes print and has to move with
# it.
LOG_OFFSET = 1e-6
LOG_OFFSET_LABEL = '1e-6'

FILE_MEAN_INDUCED = '7_exp_mean_FUS_TO_induced.tsv'
FILE_MEAN_UNINDUCED = '7_exp_mean_FUS_TO_uninduced.tsv'
REPLICATE_FILES = [('induced_rep1', 1, 'induced'), ('induced_rep2', 2, 'induced'),
                   ('uninduced_rep1', 1, 'uninduced'), ('uninduced_rep2', 2, 'uninduced')]

WT_NAME = 'positive_wildtype_clontech_7tetO_rep0'
WT_SEQUENCE = 'tccctatcagtgatagaga'          # the 19 nt tetO the library mutates
TETO_WINDOW = slice(8, 8 + 19)                # where that tetO sits in the 252 nt construct

FIG_STEM_BARPLOT = 'S04_barplot_mutation_EXPR_LEAK_FC'
FIG_STEM_COMB = 'S04_comb_observed_vs_additive'
FIG_STEM_VOLCANO = 'S04_volcano_single_mutation_fUS-Tet-on'

# the three quantities: the mean-table column they live in, the short name the barplot uses,
# the long name the combination scatter uses, and whether a higher value is the good one. The
# third one is a fold change rather than a level, so its column is derived from the other two
# (see add_lfc_column) and its values are ratios of mpra means, offset the same way.
QUANTITY_EXPR = ('induced_mpra_means', 'EXPR', 'EXPRESSION', True)
QUANTITY_LEAK = ('uninduced_mpra_means', 'LEAK', 'LEAKINESS', False)
QUANTITY_LFC = ('FC_ratio', 'LFC', 'LFC', True)

# The two lists of single substitutions the published combination heatmaps were built from, in
# the order the original wrote them out as `patterns`. No figure reads them: the panels select
# their own substitutions per position (select_single_mutations), and these two are kept so the
# run log can check that the selection still lands on the published lists.
PUBLISHED_SINGLE_LISTS = {
    'EXPR': ['1A', '2G', '3T', '4T', '5A', '6T', '7A', '8A', '9G', '10T', '11C', '12T',
             '13T', '14A', '15T', '16A', '17G', '18T', '19T'],
    'LEAK': ['1C', '2T', '3A', '4A', '5C', '6T', '7A', '8T', '9T', '10A', '11G', '12T',
             '13C', '14A', '15C', '16C', '17C', '18A', '19G'],
}

BAR_COLOR = 'tab:blue'
BAR_WIDTH = 0.7
BAR_JITTER = 0.22                             # half the point spread, in x units
POINT_SIZE = 22
BARPLOT_FIGSIZE = (17, 5.5)
WT_LINE_STYLE = {'color': 'black', 'linewidth': 1.2, 'linestyle': '--'}

# the original's combination palette: teal is worse than the wild type, tan is better
COMB_BETTER_COLOR = '#e5c185'
COMB_WORSE_COLOR = '#74a892'
COMB_POINT_SIZE = 34
COMB_FIGSIZE = (21, 7)                        # three panels, one per quantity
COMB_ANNOTATE_TOP = 2                         # how many pairs to label, per panel

# how far below the lower quartile of a panel's own values a point has to sit to count as an
# outlier of that panel: the IQR multiple of the usual fence
COMB_FENCE_K = 1.5

# the label offsets of the annotated points, cycled over the labels in order: the vertical
# steps differ by more than a line of text, so two labels of a nearly tied pair cannot land on
# one line. In the combination scatter the second label leads left instead, because the pairs
# worth labelling all sit in the top right corner there; the volcano labels five mutants per
# panel, so its cycle runs three vertical levels on the right before it leads left. The
# volcano's offsets are the wider set: its labels are tied to their marker by a leader line,
# and at the combination scatter's 8 points the line would be shorter than the marker's ring.
COMB_ANNOTATE_OFFSETS = [(8, 8), (-8, -18)]
VOLCANO_ANNOTATE_OFFSETS = [(12, 12), (12, -27), (12, 39), (-12, 12), (-12, -27)]

# the five labelled mutants of a volcano panel: a ring around the marker, in a grey that is
# neither of the two palette colours, and a leader line from the label to it. The ring has to
# be larger than the marker it circles, hence the scale of its area
VOLCANO_RING_COLOR = '#333333'
VOLCANO_RING_WIDTH = 1.2
VOLCANO_RING_SCALE = 2.2

VOLCANO_FIGSIZE = (18, 6)
VOLCANO_P_FLOOR = 1e-20                       # so that a q that underflows to 0 still plots
VOLCANO_ANNOTATE_TOP = 5                      # how many mutants to label, per panel
VOLCANO_ALPHA = 0.75


# --- the limma statistic of the original ------------------------------------------

def run_python_limma_logic(df, column1, column2):
    '''Mutants against the wild type (the last row), on two replicates. Kept as it was:
    log2(x + 1e-6), a variance moderated by the median row variance, and Benjamini-Hochberg
    over the mutants, with the two degrees of freedom of two replicates against two.'''
    n_mutants = len(df) - 1
    log_data = np.log2(df[[column1, column2]].copy() + 1e-6)

    row_vars = log_data.var(axis=1, ddof=1)
    global_variance_prior = np.median(row_vars.dropna())
    moderated_vars = (row_vars + global_variance_prior) / 2

    wt_mean = log_data.iloc[-1].mean()
    mutant_means = log_data.iloc[:n_mutants].mean(axis=1)
    logFC = mutant_means - wt_mean

    se = np.sqrt(moderated_vars.iloc[:n_mutants] * (1 / 2 + 1 / 2))
    t_stats = logFC / se
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=2))
    adj_p_vals = multipletests(p_vals, alpha=0.05, method='fdr_bh')[1]

    return pd.DataFrame({
        'names': df.iloc[:n_mutants]['names'].values,
        'logFC': logFC.values,
        'P.Value': p_vals,
        'adj.P.Val': adj_p_vals,
    })


# --- the frames -------------------------------------------------------------------

def read_mean_table(filename, value_column):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{filename} is missing from {DATA_DIR}")
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = ['seqs', value_column, 'names']
    return df[['seqs', 'names', value_column]]


def build_tetO_all():
    '''The frame the original read from ./results/round1_design/tetO_plots/tetO_all.csv,
    built in memory by the recipe in CLI_S8: every mutant plus the one clontech 7tetO row,
    which has to be the last row because it is the wild type the statistics compare against.
    The two columns are the mean tables as they are, without the negative-control
    calibration -- that is how the original csv was written, and it is the unit the barplot
    and the combination scatter work in.'''
    df_induced = read_mean_table(FILE_MEAN_INDUCED, 'induced_mpra_means')
    df_uninduced = read_mean_table(FILE_MEAN_UNINDUCED, 'uninduced_mpra_means')
    df_merged = pd.merge(df_induced, df_uninduced, on=['names', 'seqs'], how='inner')
    df_merged = df_merged[['seqs', 'names', 'induced_mpra_means', 'uninduced_mpra_means']]
    df_merged['lfc'] = df_merged['induced_mpra_means'] / df_merged['uninduced_mpra_means']

    df_merged = pd.concat((df_merged[df_merged['names'].str.startswith('mutant')],
                           df_merged[df_merged['names'] == WT_NAME])).reset_index(drop=True)
    df_merged['tetO'] = [seq[TETO_WINDOW] for seq in df_merged['seqs'].tolist()]

    # the original leaned on this row being last, and on the mutants coming first, without
    # checking either -- either one being wrong silently moves the reference
    if len(df_merged) < 2 or df_merged.iloc[-1]['names'] != WT_NAME:
        raise ValueError(f"{WT_NAME} is not the last row of the {len(df_merged)} row frame, "
                         f"so every statistic would be taken against the wrong reference")
    if not df_merged['names'].str.startswith('mutant').iloc[:-1].all():
        raise ValueError("the non-mutant rows are not just the wild type")
    return df_merged


def split_mutation_dfs(df):
    '''The original's split: the mutation fields are everything after the second underscore,
    so a single mutation has one field, a double mutant two and a palindromic mutant three.'''
    fields = df['names'].str.split('_').str[2:].apply(len)
    return df[fields == 1], df[fields == 2]


def label_single_mutations(single, wt_sequence):
    '''One row per single substitution, with its position, its base and the wild-type base
    it replaces.'''
    single = single.copy()
    single['position'] = single['names'].str.extract(r'_(\d+)').astype(int)
    single['mut_base'] = single['names'].str.extract(r'_\d+([A-Za-z])$',
                                                     expand=False).str.upper()
    if not single['position'].between(1, len(wt_sequence)).all():
        raise ValueError(f"positions {sorted(single['position'].unique())} fall outside the "
                         f"{len(wt_sequence)} nt wild-type tetO, so the wild-type base under "
                         f"each bar, and every x label, would be wrong")
    single['wt_base'] = single['position'].apply(lambda p: wt_sequence[p - 1].upper())

    # the label of a bar is (wild-type base, position); the mutation itself is the base the
    # library put at that position of the tetO window, so the two have to agree
    in_window = [seq[TETO_WINDOW][p - 1].upper()
                 for seq, p in zip(single['seqs'], single['position'])]
    mismatched = [name for name, base, seen in zip(single['names'], single['mut_base'], in_window)
                  if base != seen]
    if mismatched:
        print(f"[report]   [note] {len(mismatched)} mutations do not carry the base their name "
              f"claims at that position of the tetO window, first few: {mismatched[:3]}")

    counts = single.groupby('position').size()
    if counts.nunique() != 1:
        print(f"[report]   [note] the positions do not all carry the same number of "
              f"substitutions: {counts.to_dict()}")
    return single


# --- the barplot: three panels, one per quantity ----------------------------------

def add_display_columns(single, wt_expr, wt_leak):
    '''The three quantities the barplot shows, per substitution, in the caliber of item 1 of
    the header: the two levels as log2(value + e) and their ratio as log2((EXPR + e) /
    (LEAK + e)), which is the difference of the two logs. Returns the frame and the wild
    type's own value in each of the three units.'''
    single = single.copy()
    expr = single[QUANTITY_EXPR[0]].to_numpy(dtype=float)
    leak = single[QUANTITY_LEAK[0]].to_numpy(dtype=float)

    finite = np.isfinite(expr) & np.isfinite(leak)
    if not finite.all():
        raise ValueError(f"{int((~finite).sum())} of the {len(single)} substitutions have a "
                         f"missing level, so their bars would be invisible")
    negative = (expr < 0) | (leak < 0)
    if negative.any() or wt_expr < 0 or wt_leak < 0:
        raise ValueError(f"{int(negative.sum())} substitutions (and wt: {wt_expr < 0 or wt_leak < 0}) "
                         f"have a negative level, which the library's mpra means never do")
    # the levels are logged with the offset, so a level of exactly 0 is fine -- but a level at
    # or below -LOG_OFFSET would make the ratio of panel 3 meaningless
    if (leak + LOG_OFFSET <= 0).any() or wt_leak + LOG_OFFSET <= 0:
        raise ValueError("a LEAKINESS level is at or below the log offset, so the "
                         "EXPRESSION / LEAKINESS ratio would not be defined")

    single['EXPR_log2'] = np.log2(expr + LOG_OFFSET)
    single['LEAK_log2'] = np.log2(leak + LOG_OFFSET)
    single['FC_log2'] = single['EXPR_log2'] - single['LEAK_log2']
    wt = {'EXPR': np.log2(wt_expr + LOG_OFFSET),
          'LEAK': np.log2(wt_leak + LOG_OFFSET),
          'FC': np.log2(wt_expr + LOG_OFFSET) - np.log2(wt_leak + LOG_OFFSET)}
    return single, wt


def plot_mutation_barplot(single, wt):
    '''One figure, three panels: the EXPRESSION level, the LEAKINESS level and their ratio,
    one bar per position. The bar is the mean of the position's three substitutions and the
    three are drawn on top of it; the dashed line is the wild type's own value in the panel's
    unit. Everything is on the caliber of item 1 of the header, so nothing here is a ratio to
    the wild type unless the panel's own label says so.'''
    panels = (
        ('EXPR', 'EXPR_log2', QUANTITY_EXPR[1], f'log2({QUANTITY_EXPR[1]} + {LOG_OFFSET_LABEL})'),
        ('LEAK', 'LEAK_log2', QUANTITY_LEAK[1], f'log2({QUANTITY_LEAK[1]} + {LOG_OFFSET_LABEL})'),
        ('FC', 'FC_log2', f'{QUANTITY_EXPR[1]} / {QUANTITY_LEAK[1]}',
         f'log2(({QUANTITY_EXPR[1]} + {LOG_OFFSET_LABEL}) / ({QUANTITY_LEAK[1]} + {LOG_OFFSET_LABEL}))'),
    )
    positions = sorted(single['position'].unique())
    x_labels = [f"{single.loc[single['position'] == p, 'wt_base'].iloc[0]}{p}" for p in positions]

    fig, axes = plt.subplots(1, len(panels), figsize=BARPLOT_FIGSIZE)
    wt_handle = None
    for ax, (key, column, name, y_label) in zip(axes, panels):
        grouped = single.groupby('position')[column]
        bar_mean = grouped.mean().reindex(positions)
        ax.bar(range(len(positions)), bar_mean.values, width=BAR_WIDTH, color=BAR_COLOR,
               edgecolor='k')
        for i, position in enumerate(positions):
            points = single.loc[single['position'] == position, column].to_numpy()
            ax.scatter(i + np.linspace(-BAR_JITTER, BAR_JITTER, len(points)), points,
                       s=POINT_SIZE, color='black', edgecolors='white', linewidths=0.4, zorder=3)
        line = ax.axhline(y=wt[key], label='WT', **WT_LINE_STYLE)
        wt_handle = wt_handle or line

        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(x_labels, rotation=90, fontsize=10)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_xlabel('Position (WT base)', fontsize=12)

        print(f"[report] {name}: WT {wt[key]:+.2f} in the plotted unit; bars are the mean of "
              f"{int(grouped.size().min())}..{int(grouped.size().max())} substitutions per "
              f"position; {column} runs {single[column].min():+.2f} .. "
              f"{single[column].max():+.2f}; {int((bar_mean > wt[key]).sum())} of "
              f"{len(positions)} positions sit above WT")

    # one legend for the figure: the three panels share the one dashed line, and repeating it
    # would push the panel labels around in each of them
    # it goes above the figure rather than inside it: at the top of the figure the legend
    # would sit on the tallest bars, and tight_layout has no room for it either
    fig.legend([wt_handle], ['WT'], loc='lower center', bbox_to_anchor=(0.5, 1.0),
               ncol=1, frameon=False, fontsize=11)
    plt.tight_layout()
    for fmt in ('png', 'svg', 'pdf'):
        plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM_BARPLOT}.{fmt}'), format=fmt,
                    dpi=DPI, bbox_inches='tight')
    plt.close()


# --- the combination: observed against the additive prediction --------------------

def add_lfc_column(frame):
    '''The induced / uninduced fold change of a row of the mean table, in the linear unit whose
    log2 the barplot's third panel draws and whose ratio to the wild type's the LFC panel of
    the combination scatter measures. Copied rather than written through, like the other frame
    helpers here, so that the caller's frame keeps its columns.'''
    expr = frame[QUANTITY_EXPR[0]].to_numpy(dtype=float)
    leak = frame[QUANTITY_LEAK[0]].to_numpy(dtype=float)
    finite = np.isfinite(expr) & np.isfinite(leak)
    if not finite.all():
        raise ValueError(f"{int((~finite).sum())} of the {len(frame)} rows have a missing level, "
                         f"so their induced / uninduced fold change would be a hole in the "
                         f"scatter")
    if (expr + LOG_OFFSET <= 0).any() or (leak + LOG_OFFSET <= 0).any():
        raise ValueError("an EXPRESSION or LEAKINESS level is at or below the log offset, so "
                         "the induced / uninduced fold change would be negative or undefined")
    frame = frame.copy()
    frame[QUANTITY_LFC[0]] = (expr + LOG_OFFSET) / (leak + LOG_OFFSET)
    return frame


def select_single_mutations(single):
    '''The one substitution per position of the tetO that each combination panel is built from:
    the highest EXPRESSION, the lowest LEAKINESS, the highest induced / uninduced fold change.
    The original wrote the first two out as a hard-coded list of `patterns`; deriving them here
    keeps all three panels to one rule and gives the third quantity a list of its own. Returns
    {short name: [mutation names]}, each list in position order.'''
    picks = {}
    for value_column, short_name, long_name, better_is_higher in (QUANTITY_EXPR, QUANTITY_LEAK,
                                                                  QUANTITY_LFC):
        chosen, tied = [], []
        for position, group in single.groupby('position'):
            # the name breaks a tie, so the pick does not depend on the frame's row order
            ranked = group.sort_values([value_column, 'mut_base'],
                                       ascending=[not better_is_higher, True])
            chosen.append(f"{int(position)}{ranked.iloc[0]['mut_base']}")
            if len(ranked) > 1 and ranked[value_column].iloc[0] == ranked[value_column].iloc[1]:
                tied.append((int(position), list(ranked['mut_base'])))
        picks[short_name] = chosen
        print(f"[report] {long_name} panel takes one substitution per position: "
              f"{' '.join(chosen)}")
        if tied:
            print(f"[report]   [note] {len(tied)} positions have the best two substitutions tied "
                  f"for {long_name}, so the pick falls back on the base: {tied}")

    # the two published lists are the check that this selection still lands on the figure that
    # was published, which is the whole reason they are in the file at all
    for short_name, published in PUBLISHED_SINGLE_LISTS.items():
        derived = picks[short_name]
        differing = [(a, b) for a, b in zip(derived, published) if a != b]
        if len(derived) == len(published) and not differing:
            print(f"[report]   the derived {short_name} list is the published one, position by "
                  f"position")
        else:
            print(f"[report]   [note] the derived {short_name} list differs from the published "
                  f"one ({len(derived)} vs {len(published)} positions, first differences "
                  f"{differing[:3]})")
    return picks


def retrieve_double_mutations(double_mutation_df, single_mutations):
    '''Kept as it was: a double mutant is kept when both of its substitutions are in the
    given list of single mutations.'''
    filtered_rows = []
    for _, row in double_mutation_df.iterrows():
        mutations = row['names'].split('_')[2:]
        if len(mutations) == 2 and mutations[0] in single_mutations \
                and mutations[1] in single_mutations:
            filtered_rows.append(row)
    result_df = pd.DataFrame(filtered_rows)
    if 'seqs' in result_df.columns:
        result_df = result_df.drop('seqs', axis=1)
    return result_df.reset_index(drop=True)


def generate_mutation_order(mutation_dict):
    '''Position first, then base, as in the original.'''
    def mutation_key(mutation):
        return (int(''.join(filter(str.isdigit, mutation))), mutation[-1])
    return sorted(mutation_dict, key=mutation_key)


def single_effects(single, quantity, wt_value):
    '''The effect of each single substitution on the "+ = better than the wild type" scale,
    keyed the way the combination names write it ("1C"), so that a pair's prediction can be
    the plain sum of its two members. The sign convention is the one the scatter plots.'''
    value_column, _, long_name, better_is_higher = quantity
    values = single[value_column].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or (values < 0).any() or wt_value < 0:
        raise ValueError(f"a {long_name} level is missing or negative, so its effect against "
                         f"the wild type would not be defined")
    effect = np.log2((values + LOG_OFFSET) / (wt_value + LOG_OFFSET))
    if not better_is_higher:
        effect = -effect
    return {f"{int(p)}{base}": float(e)
            for p, base, e in zip(single['position'], single['mut_base'], effect)}


def comb_window(observed, predicted):
    '''The lower and upper bound of one panel's effect axis. A pair whose induced mean has
    collapsed sits at log2 of a ratio near zero -- the bottom of the EXPRESSION panel is at -8
    -- and the few dozen of them stretch the axis until the bulk of the library is a line;
    item 5 of the header has the reading. The bound stops at the outlier fence of the panel's
    own values, pooled over both of its axes, so the pairs below it can be pinned to the edge.
    Pooling the axes is what keeps the fence from cutting into LEAKINESS, where the predicted
    effects spread the quartiles far wider than the observed ones: there the fence lands below
    everything drawn, the bound comes out as the data range and nothing is pinned. The two
    bounds are then widened into the panel's one interval, which both of its axes carry.'''
    pooled = np.concatenate([observed, predicted])
    lower_quartile, upper_quartile = np.percentile(pooled, [25, 75])
    fence = float(lower_quartile - COMB_FENCE_K * (upper_quartile - lower_quartile))
    return max(float(observed.min()), fence), float(observed.max())


def plot_comb_additivity(comb_df, quantity, wt_value, effects, ax):
    '''One panel per quantity: x is the sum of the two single substitutions' effects, y is the
    pair's own effect, both on the "+ = better than the wild type" scale, and the dashed line
    is the additive model. A pair below the line improves less than its two members added up
    -- which is where this library sits -- and the tan points are the pairs already better
    than the wild type. Pairs too far below the wild type to fit the window are drawn as
    triangles on its lower edge.'''
    value_column, short_name, long_name, better_is_higher = quantity

    pairs, predictions, observed = [], [], []
    for _, row in comb_df.iterrows():
        first, second = row['names'].split('_')[2:]
        for mutation in (first, second):
            if mutation not in effects:
                raise ValueError(f"the pair {first}+{second} contains {mutation}, which is not "
                                 f"one of the single substitutions, so its additive prediction "
                                 f"would be a guess")
        effect = np.log2((row[value_column] + LOG_OFFSET) / (wt_value + LOG_OFFSET))
        if not better_is_higher:
            effect = -effect
        pairs.append(f'{first}+{second}')
        predictions.append(effects[first] + effects[second])
        observed.append(float(effect))

    predictions = np.array(predictions)
    observed = np.array(observed)
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predictions)):
        raise ValueError(f"a {long_name} pair has a missing level, so its effect would be a "
                         f"hole in the scatter")

    better = observed > 0
    ax.scatter(predictions[better], observed[better], s=COMB_POINT_SIZE,
               color=COMB_BETTER_COLOR, edgecolors='black', linewidths=0.5,
               label='better than WT', zorder=3)
    ax.scatter(predictions[~better], observed[~better], s=COMB_POINT_SIZE,
               color=COMB_WORSE_COLOR, edgecolors='black', linewidths=0.5,
               label='worse than WT', zorder=3)

    # both axes carry one interval -- the union of what the two of them have to show, and the
    # same interval in all three panels' two axes. Forced apart (each axis tight around its own
    # data, which is what the first version did) the model line tilts away from 45 degrees and
    # the cloud is squeezed into a band: the LEAKINESS panel drew 5.8 units of prediction
    # against 2.7 of effect, so its bulk sat in the lower third, and the LFC panel had the
    # opposite tilt. Item 5 of the header has the reading
    x_lo, x_hi = float(predictions.min()), float(predictions.max())
    x_pad = 0.05 * (x_hi - x_lo) if x_hi > x_lo else 1.0
    y_lo, y_hi = comb_window(observed, predictions)
    y_pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 1.0
    low = min(x_lo - x_pad, y_lo - y_pad)
    high = max(x_hi + x_pad, y_hi + y_pad)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)

    grid = np.array([low, high])
    ax.plot(grid, grid, linestyle='--', color='black', linewidth=1.0,
            label='additive model (y = x)', zorder=2)
    if low <= 0 <= high:
        ax.axhline(y=0, label='WT (no change)', **WT_LINE_STYLE)

    # the pairs the window does not reach keep their x, but are drawn on the edge of the panel:
    # the trim is about which points set the scale, not about dropping any of them, so they are
    # still on the figure and the corner says how many and how deep. Their colour still says
    # which side of the wild type they sit on
    pinned = observed < y_lo
    for mask, colour in ((pinned & better, COMB_BETTER_COLOR),
                         (pinned & ~better, COMB_WORSE_COLOR)):
        if mask.any():
            ax.scatter(predictions[mask], np.full(int(mask.sum()), y_lo), s=COMB_POINT_SIZE,
                       marker='v', color=colour, edgecolors='black', linewidths=0.5, zorder=3)
    if pinned.any():
        # the count and the depth of the pinned pairs go into the legend and not into a corner
        # of the panel: the corners here hold the legend itself and the row of triangles, and a
        # free text note lands on one of the two whichever corner it is given
        ax.scatter([], [], marker='v', s=COMB_POINT_SIZE,
                   color=COMB_WORSE_COLOR if (pinned & ~better).any() else COMB_BETTER_COLOR,
                   edgecolors='black', linewidths=0.5,
                   label=f'{int(pinned.sum())} of {len(pairs)} pairs below the window, '
                         f'min {observed.min():+.2f}')

    # the best pairs, which is what one would build next; the lead keeps the labels off the
    # points
    top = np.argsort(-observed)[:COMB_ANNOTATE_TOP]
    for rank, index in enumerate(top):
        dx, dy = COMB_ANNOTATE_OFFSETS[rank % len(COMB_ANNOTATE_OFFSETS)]
        ax.annotate(pairs[index], xy=(predictions[index], observed[index]),
                    xytext=(dx, dy), textcoords='offset points', fontsize=9, zorder=4,
                    ha='left' if dx > 0 else 'right')

    sign = '+' if better_is_higher else '-'
    ax.set_xlabel(f'sum of the two single effects\n{sign}log2({long_name} single / WT), '
                  f'+ = better', fontsize=12)
    ax.set_ylabel(f'pair effect, {sign}log2({long_name} pair / WT), + = better', fontsize=12)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)

    slope = float(np.polyfit(predictions, observed, 1)[0])
    deficit = predictions - observed
    print(f"[report] {long_name}: {len(pairs)} pairs kept; observed effect "
          f"{observed.min():+.2f} .. {observed.max():+.2f}, additive prediction "
          f"{predictions.min():+.2f} .. {predictions.max():+.2f}; "
          f"{int(better.sum())} of {len(pairs)} pairs are better than WT; "
          f"correlation {np.corrcoef(predictions, observed)[0, 1]:.3f}, slope "
          f"{slope:.3f}, median (prediction - observed) {np.median(deficit):+.2f} "
          f"({int((deficit > 0).sum())} of {len(pairs)} pairs below the line); best pair "
          f"{pairs[int(np.argmax(observed))]} at {observed.max():+.2f} while the model "
          f"predicts {predictions[int(np.argmax(observed))]:+.2f}")
    return ax


def make_combination_figure(comb_dfs):
    '''The three quantities side by side, one shared reading.'''
    fig, axes = plt.subplots(1, len(comb_dfs), figsize=COMB_FIGSIZE)
    for ax, (comb_df, quantity, wt_value, effects) in zip(axes, comb_dfs):
        plot_comb_additivity(comb_df, quantity, wt_value, effects, ax)
    plt.tight_layout()
    for fmt in ('png', 'svg', 'pdf'):
        plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM_COMB}.{fmt}'), format=fmt, dpi=DPI,
                    bbox_inches='tight')
    plt.close()


# --- the volcano: three statistics, effect size against significance ---------------

def plot_mutation_volcano(df, hit_names):
    '''One panel per statistic of the original pipeline: x is the log fold change against the
    wild type, y is -log10 of its adjusted p-value, so the significant mutants are the ones
    high up and the strong ones are the ones far from the dashed line at 0. The colours carry
    the direction of the effect, tan for the better-than-wild-type side of each statistic, and
    nothing else is drawn on the markers: the mutants worth building are named in text, five
    per panel, because there is no outline left to recognise them by.'''
    panels = (
        ('logFC_induced', 'adj.P.Val_induced', 'EXPRESSION', True),
        ('logFC_uninduced', 'adj.P.Val_uninduced', 'LEAKINESS', False),
        ('logFC_lfc', 'adj.P.Val_lfc', 'induction trend', True),
    )

    fig, axes = plt.subplots(1, len(panels), figsize=VOLCANO_FIGSIZE)
    q_threshold = -np.log10(0.05)
    for ax, (fc_column, q_column, name, better_is_higher) in zip(axes, panels):
        log_fc = df[fc_column].to_numpy(dtype=float)
        q_values = df[q_column].to_numpy(dtype=float)
        finite = np.isfinite(log_fc) & np.isfinite(q_values)
        if not finite.all():
            raise ValueError(f"{int((~finite).sum())} of the {len(df)} mutants have no "
                             f"{name} statistic, so the volcano would quietly lose them")
        if (q_values < 0).any():
            raise ValueError(f"an adjusted p-value of {name} is negative")
        floored = q_values < VOLCANO_P_FLOOR
        if floored.any():
            print(f"[report]   [note] {int(floored.sum())} {name} adjusted p-values underflow "
                  f"to 0 and are drawn at the {VOLCANO_P_FLOOR:g} floor")
        y_values = -np.log10(np.clip(q_values, VOLCANO_P_FLOOR, None))

        labels = df['names'].str.split('_').str[2]
        better = (log_fc > 0) if better_is_higher else (log_fc < 0)
        # one legend explains all three panels, so the two colour entries are named after the
        # reading they carry rather than after one panel's quantity: tan is the better side of
        # each statistic, whichever sign that is
        ax.scatter(log_fc[better], y_values[better], s=COMB_POINT_SIZE, alpha=VOLCANO_ALPHA,
                   color=COMB_BETTER_COLOR, label='better than WT', zorder=3)
        ax.scatter(log_fc[~better], y_values[~better], s=COMB_POINT_SIZE, alpha=VOLCANO_ALPHA,
                   color=COMB_WORSE_COLOR, label='worse than WT', zorder=3)

        candidates = df['names'].isin(hit_names).to_numpy()
        if not candidates.any():
            raise ValueError(f"none of the {len(hit_names)} mutations that raise EXPRESSION, "
                             f"lower LEAKINESS and raise the induction trend are in the {name} "
                             f"frame, so there would be nothing to label")

        ax.axvline(x=0, **WT_LINE_STYLE)
        ax.axhline(y=q_threshold, color='black', linewidth=1.0, linestyle=':')
        ax.text(0.02, q_threshold + 0.15, 'q = 0.05', transform=ax.get_yaxis_transform(),
                fontsize=9, va='bottom', ha='left')

        # the five mutants of this panel that it is most sure about, out of the candidates every
        # panel shares; labelling all nine in all three panels would bury the points under text.
        # The Benjamini-Hochberg step clamps a whole block of mutants to the same adjusted
        # p-value, so the ranking is broken by the size of the effect -- left to the sort, which
        # five of the tied block get labelled is arbitrary
        rank = np.lexsort((-np.abs(log_fc), -y_values))
        labelled = [index for index in rank if candidates[index]][:VOLCANO_ANNOTATE_TOP]
        # the five are ringed, and each label is tied to its own ring by a leader line: with no
        # edge on the markers at all, the name alone leaves the reader to guess which of two
        # neighbouring points it was written for
        ax.scatter(log_fc[labelled], y_values[labelled],
                   s=COMB_POINT_SIZE * VOLCANO_RING_SCALE, facecolors='none',
                   edgecolors=VOLCANO_RING_COLOR, linewidths=VOLCANO_RING_WIDTH,
                   label='_nolegend_', zorder=4)
        for order, index in enumerate(labelled):
            dx, dy = VOLCANO_ANNOTATE_OFFSETS[order % len(VOLCANO_ANNOTATE_OFFSETS)]
            ax.annotate(str(labels.iloc[index]), xy=(log_fc[index], y_values[index]),
                        xytext=(dx, dy), textcoords='offset points', fontsize=9, zorder=5,
                        ha='left' if dx > 0 else 'right',
                        arrowprops=dict(arrowstyle='-', color=VOLCANO_RING_COLOR,
                                        linewidth=0.8, shrinkA=1, shrinkB=6))

        ax.set_xlabel(f'logFC({name}) vs WT, + = '
                      f'{"higher" if better_is_higher else "lower"} than WT', fontsize=12)
        ax.set_ylabel(f'-log10(adjusted p), {name}', fontsize=12)

        print(f"[report] volcano {name}: logFC {log_fc.min():+.2f} .. {log_fc.max():+.2f}, "
              f"-log10(q) {y_values.min():.2f} .. {y_values.max():.2f}; q < 0.05 in "
              f"{int((q_values < 0.05).sum())} of {len(df)}, "
              f"{int(((q_values < 0.05) & better).sum())} of them on the better side; labelled "
              f"{', '.join(str(labels.iloc[index]) for index in labelled)} at adjusted p "
              f"{', '.join(f'{q_values[index]:.4f}' for index in labelled)}")

    # the same two colours in every panel, so one legend explains all three
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 1.0),
               ncol=len(legend_labels), frameon=False, fontsize=11)
    plt.tight_layout()
    for fmt in ('png', 'svg', 'pdf'):
        plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM_VOLCANO}.{fmt}'), format=fmt, dpi=DPI,
                    bbox_inches='tight')
    plt.close()


# --- run: the mean-table frame and the barplot ------------------------------------

os.makedirs(FIG_DIR, exist_ok=True)

df_all = build_tetO_all()
single_mutations, double_mutations = split_mutation_dfs(df_all)
single_mutations = label_single_mutations(single_mutations, WT_SEQUENCE)
wt_row = df_all.iloc[-1]
print(f"[report] tetO_all frame built in memory: {len(df_all)} rows, wild type {WT_NAME}; "
      f"EXPRESSION {wt_row['induced_mpra_means']:.3f}, "
      f"LEAKINESS {wt_row['uninduced_mpra_means']:.3f} (mean-table units)")
print(f"[report] {len(single_mutations)} single substitutions, "
      f"{len(double_mutations)} double mutants")

single_mutations, wt_display = add_display_columns(single_mutations,
                                                   wt_row['induced_mpra_means'],
                                                   wt_row['uninduced_mpra_means'])
plot_mutation_barplot(single_mutations, wt_display)
