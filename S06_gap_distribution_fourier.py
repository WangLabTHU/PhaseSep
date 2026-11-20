import os
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'

# Project layout: data in ./data, figures in ./figs (checkpoints, where used, in ./checks)
DATA_DIR = './data'
FIG_DIR = './figs'
DPI = 300

# The four round-1 mean tables under the renamed project file names: the FUS-rtTA arm
# (7_exp_mean_induced / _uninduced) and the rtTA arm (7_exp_mean_E_induced / _uninduced).
FILE_FUS_INDUCED = '7_exp_mean_FUS_TO_induced.tsv'
FILE_FUS_UNINDUCED = '7_exp_mean_FUS_TO_uninduced.tsv'
FILE_TO_INDUCED = '7_exp_mean_TO_induced.tsv'
FILE_TO_UNINDUCED = '7_exp_mean_TO_uninduced.tsv'

FIG_STEM_DISTRIBUTION = 'S06_gap_distribution_EXPR_LEAK'
FIG_STEM_FOURIER = 'S06_gap_fourier_EXPR_LEAK'

# The offset every log2 in this script takes: a level is log2(value + LOG_OFFSET). It is
# 1e-6, the same offset the limma statistic of the rest of the project uses, so every figure
# shares one convention. The original drew its levels raw; LOG_OFFSET_LABEL is what the axes
# print and the two lines move together.
LOG_OFFSET = 1e-6
LOG_OFFSET_LABEL = '1e-6'

# Deviations from refs_dqx/fig3b_spacer_fourier.py, the script the two
# 4.1_distribution_across_gap_EXPR_LEAK.png / 4.2_fourier_across_gap_EXPR_LEAK.png figures
# come from. It is the only implementation of either figure in the two reference trees:
# refs_pmn/Figure_5/F5_syntax_activity.py draws the same 6tetO curve from the same lookahead
# pattern, but as a two-panel induction / leakiness figure, which is not these figures. The
# design selection, the negative-control calibration and the Fourier recipe are carried over
# line for line.
#   1. the two levels are drawn as log2(value + LOG_OFFSET) (requested 2026-09-29, standing
#      instruction for the whole project; the levels here are the same calibrated
#      induced/uninduced means S05, S06, S07 and S08 draw, so the four figures agree). The
#      original drew the calibrated means raw. Three consequences, all of them forced by the
#      transform rather than chosen:
#        a. the error bar is symmetric (mean +- SD). The original clipped the lower bar at 0
#           -- mean - min(mean - SD, 0) -- because a raw expression mean cannot be negative,
#           which stops being true of a log2 level.
#        b. the floor every panel was pinned to, ylim(bottom=-0.1), is the floor of the raw
#           axis. A log2 level can sit below it, so the bottom is now the original's -0.1 or
#           the lowest drawn value, whichever is lower; a panel whose values all stay above
#           -0.1 keeps the original's exact bottom (see panel_bottom).
#        c. the values a log2 cannot be taken of -- not finite, or at most -LOG_OFFSET -- are
#           dropped and counted (see gap_curve) instead of being drawn as holes.
#   2. png, svg and pdf are all written and the figure is closed. The original wrote the png
#      only, called plt.show() and never closed the figure.
#   3. the Fourier amplitude axis is therefore the transform of the same levels, which is
#      printed on the axis. A monotone transform moves the amplitude profile but not the
#      position of a peak, so the annotated periods are the original's.
#   4. the layout is the original's, including the per-column y label, the widths
#      [8, 8, 8, 6, 4, 3] and the sizes, with these changes requested on 2026-09-29:
#        a. the background grid is off. The original turned it on in the first figure
#           (ax.grid(True, which='both', axis='both')) and had it commented out in the
#           second; both are now off.
#        b. the twelve per-panel legends are replaced by one shared legend per figure, drawn
#           above the canvas. The original put a legend inside every panel, where it landed
#           on the curves of the wide panels. Same move S07 carries.
#        c. the peak period labels no longer overlap. The original wrote each one at the peak
#           plus a fixed 3% of the panel maximum, so peaks a few gap units apart, and the two
#           arms peaking at the same frequency, printed on top of each other. They are now
#           laid out on levels (see stack_peak_labels) and the axis is given the room.
#        d. the labels no longer run into the guide of a peak or into the curves. The
#           original drew every guide the full height of the panel, which ran it through the
#           digits of its own label, and put every label one fixed step above its peak, which
#           in a flat stretch left it on the markers. A guide now stops below every label
#           whose box it would pass under, and a label is raised until it clears both the
#           curves and the markers drawn under it.
#        e. the labels are 9 point instead of 7, which is what the level spacing, the
#           clearance and the axis room are all computed from.
#        f. the Fourier amplitude axis starts at -0.5 instead of 0, the original's floor.

# --- layout and style, the original's numbers ---------------------------------

TETO_RANGE = list(range(2, 8))
COL_WIDTHS = [8, 8, 8, 6, 4, 3]
DISTRIBUTION_FIGSIZE = (36, 10)
FOURIER_FIGSIZE = (30, 10)
GRID_HEIGHT_RATIOS = [1, 1]
GRID_HSPACE = 0.15
GRID_WSPACE = 0.25

# row 0 of both grids is the induced (EXPR) level, row 1 the uninduced (LEAK) level
GAP_PANELS = [('induced', 'EXPR'), ('uninduced', 'LEAK')]

FUS_COLOR = 'red'
TETR_COLOR = 'blue'
FUS_STYLE = '-x'
TETR_STYLE = '-o'
FUS_LABEL = 'FUS-Tet-on'
TETR_LABEL = 'Tet-on'
CAPSIZE = 5
TITLE_FONTSIZE = 11
AXIS_LABEL_FONTSIZE = 10
LEGEND_FONTSIZE = 8
PEAK_FONTSIZE = 9
PEAK_LINE_ALPHA = 0.4
X_LABEL_GAP = 'Gap unit (bp)'
X_LABEL_FREQ = 'Frequency (1/gap unit)'

# one shared legend per figure, drawn above the canvas instead of inside every panel
LEGEND_ANCHOR = (0.5, 1.0)

# where the Fourier amplitude axis starts
FOURIER_YLIM_BOTTOM = -0.5

# stacking the peak period labels. One level is this many label heights tall, a character of
# the label font is this wide, neighbouring labels keep this fraction of the axis between
# them, this many label heights of blank space are left above the highest label, and the
# layout is repeated at most this many times while the axis grows to hold it.
PEAK_LANE_ASCENT = 1.5
PEAK_LANE_CHAR_EM = 0.55
PEAK_LANE_X_GAP = 0.01
PEAK_LANE_HEADROOM = 1.2
PEAK_LANE_PASSES = 6
# a label keeps this many label heights clear of the curves drawn under it, and the dashed
# guide of a peak stops this many label heights below every label it would otherwise run into
PEAK_LABEL_CLEARANCE = 0.5
PEAK_LINE_TO_LABEL_GAP = 0.25

# the arms are drawn with the default marker, and a label has to clear the marker and not
# just the point it sits on, so the marker is measured here and added to that clearance
MARKER_RADIUS_INCH = 0.5 * plt.rcParams['lines.markersize'] / 72.0

# the five '_'-separated fields the original read a spacer name into, in order
GAP_NAME_FIELDS = ['group', 'motif number', 'gap', 'flanking', 'replicate']


# --- data ---------------------------------------------------------------------

def read_mean_table(filename, value_column, label):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{filename} is missing from {DATA_DIR} (the {label} mean table)")
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = ['seqs', value_column, 'names']
    return df[['seqs', 'names', value_column]]


def calibrate_arm(df, label):
    '''Divide the two mean columns and the fold change by the negative-control mean of the
    same column, as in the original. The fold-change calibration is the mean of the
    per-sequence ratios, which is not the ratio of the two mean calibrations; both are left
    exactly as they were. The fold change is calibrated even though neither figure here
    draws it, so that this block stays the one the other scripts carry.'''
    columns = ['induced_mpra_means', 'uninduced_mpra_means', 'lfc']
    cali = df.loc[df['names'].str.contains('negative_random'), columns].mean()

    # the original divided by these without checking: one missing negative_random row, or a
    # calibration of zero, turns every calibrated value into NaN
    for column in columns:
        if not np.isfinite(cali[column]) or cali[column] <= 0:
            raise ValueError(
                f"the negative_random rows of the {label} arm give a calibration of "
                f"{cali[column]} for {column}, so the whole arm would come out as NaN; the "
                f"original divided by it without checking")

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
          f"induced {cali['induced_mpra_means']:.4f} | uninduced {cali['uninduced_mpra_means']:.4f} "
          f"| fold change {cali['lfc']:.4f}")
    return merged


def extract_gap(name):
    '''The gap of one design, in bp, from its name. The original's own reader, used by the
    Fourier half; the panel half reads the same number off the split name field instead.'''
    match = re.search(r'(\d+)gap', str(name))
    return int(match.group(1)) if match else np.nan


def select_gap_frame(motif_num):
    '''The designs of one panel of the first grid. The lookahead pattern is the original's:
    a name carrying both "<n>tetO" and "gap". It is not anchored, so the names it pulls in
    are checked against their own "motif number" field and against the reader the Fourier
    half uses, and anything that disagrees is reported.'''
    pattern = rf'(?=.*{motif_num}tetO)(?=.*gap)'
    frame = df_merged[df_merged['names'].str.contains(pattern, regex=True)].copy()
    if frame.empty:
        raise ValueError(f"no design name carries both '{motif_num}tetO' and 'gap'")

    label = frame['names'].str.split('_', expand=True)
    if label.shape[1] != len(GAP_NAME_FIELDS) or label.isna().any().any():
        offenders = frame.loc[label.isna().any(axis=1), 'names'].head(5).tolist()
        raise ValueError(
            f"a design name of the {motif_num}tetO panel does not split into "
            f"{len(GAP_NAME_FIELDS)} '_'-separated fields, so the original's five field "
            f"names would not fit it: {offenders}")
    label.columns = GAP_NAME_FIELDS
    frame = pd.concat([frame, label], axis=1)

    gap_field = frame['gap'].str.extract(r'(\d+)', expand=False)
    if gap_field.isna().any():
        offenders = frame.loc[gap_field.isna(), 'names'].head(5).tolist()
        raise ValueError(f"the 'gap' field of {int(gap_field.isna().sum())} designs of the "
                         f"{motif_num}tetO panel carries no number: {offenders}")
    frame['gap_num'] = gap_field.astype(int)

    # the pattern is not anchored, so a name whose own tetO count is not this panel's would
    # be pooled into it silently; so would a name the two readers disagree about
    off_teto = frame.loc[frame['motif number'] != f'{motif_num}tetO', 'names']
    if len(off_teto):
        print(f"[report]   [note] ntetO {motif_num}: the unanchored pattern also pulls in "
              f"{len(off_teto)} designs whose own field is not {motif_num}tetO "
              f"({off_teto.head(3).tolist()}); they are drawn in this panel as the original "
              f"drew them")
    regex_gap = frame['names'].apply(extract_gap)
    mismatch = int((regex_gap != frame['gap_num']).sum())
    if mismatch:
        print(f"[report]   [note] ntetO {motif_num}: {mismatch} designs are read as one gap "
              f"by the split field and another by the regex, so the two grids do not pool "
              f"the same designs")
    return frame


def gap_curve(frame, level, arm):
    '''The mean and SD of log2(value + LOG_OFFSET) per gap, for one level of one arm.
    Values that no log2 can be taken of are dropped and counted, rather than being drawn as
    holes; a gap whose designs are all dropped leaves no point at all.'''
    column = f'{level}_mpra_means_{arm}'
    values = np.asarray(frame[column], dtype=float)
    drawable = np.isfinite(values) & (values > -LOG_OFFSET)
    dropped = int((~drawable).sum())

    drawn = frame.loc[drawable, ['gap_num']].copy()
    drawn['value'] = np.log2(values[drawable] + LOG_OFFSET)
    grouped = drawn.groupby('gap_num')['value'].agg(['mean', 'std', 'count']).reset_index()
    grouped = grouped.sort_values('gap_num').reset_index(drop=True)
    return grouped, dropped


def panel_bottom(values):
    '''The original pinned every gap panel to ylim(bottom=-0.1). That is the floor of a raw
    expression axis; it is kept whenever nothing is drawn below it, otherwise the floor
    moves just under the lowest drawn value.'''
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -0.1
    low = float(values.min())
    if low >= -0.1:
        return -0.1
    span = float(values.max()) - low
    return low - (0.05 * span if span > 0 else 0.1)


os.makedirs(FIG_DIR, exist_ok=True)

df_FUS_Trans_merged = build_arm('FUS-rtTA', FILE_FUS_INDUCED, FILE_FUS_UNINDUCED)
df_tetR_Trans_merged = build_arm('rtTA', FILE_TO_INDUCED, FILE_TO_UNINDUCED)

df_merged = pd.merge(df_FUS_Trans_merged, df_tetR_Trans_merged, on=['seqs', 'names'],
                     suffixes=('_FUS', '_tetR'))
print(f"[report] both arms merged on (names, seqs): {len(df_merged)} designs")
print("[report] spacer designs per tetO count, from the Fourier half's own prefix filter:")
for n in TETO_RANGE:
    frame = df_merged[df_merged['names'].str.startswith(f'spacer_{n}tetO')]
    gaps = frame['names'].apply(extract_gap).dropna()
    if frame.empty or gaps.empty:
        print(f"[report]   {n}tetO: {len(frame)} designs, no gap to read")
        continue
    print(f"[report]   {n}tetO: {len(frame)} designs, {int(gaps.nunique())} distinct gaps "
          f"[{int(gaps.min())} .. {int(gaps.max())}]")


# --- figure 1: the level of each gap, per tetO count ------------------------------------------

fig = plt.figure(figsize=DISTRIBUTION_FIGSIZE)
gs = GridSpec(2, len(TETO_RANGE), height_ratios=GRID_HEIGHT_RATIOS, width_ratios=COL_WIDTHS,
              hspace=GRID_HSPACE, wspace=GRID_WSPACE)

legend_handles = None
for i, motif_num in enumerate(TETO_RANGE):
    frame = select_gap_frame(motif_num)
    print(f"[report] ntetO {motif_num}: {len(frame)} designs, gaps "
          f"[{int(frame['gap_num'].min())} .. {int(frame['gap_num'].max())}]")

    for row_idx, (level, panel_name) in enumerate(GAP_PANELS):
        ax = fig.add_subplot(gs[row_idx, i])
        drawn = []
        for arm, style, color, arm_label in (('FUS', FUS_STYLE, FUS_COLOR, FUS_LABEL),
                                             ('tetR', TETR_STYLE, TETR_COLOR, TETR_LABEL)):
            grouped, dropped = gap_curve(frame, level, arm)
            if dropped:
                print(f"[report]   [note] ntetO {motif_num} {panel_name} {arm}: {dropped} "
                      f"designs have no log2(value + {LOG_OFFSET_LABEL}) and are not drawn")
            singleton = int((grouped['count'] < 2).sum())
            if singleton:
                print(f"[report]   [note] ntetO {motif_num} {panel_name} {arm}: {singleton} "
                      f"gaps hold a single design, so their SD is undefined")
            if grouped.empty:
                print(f"[report]   [note] ntetO {motif_num} {panel_name} {arm}: no gap is "
                      f"drawable")
                continue
            ax.errorbar(grouped['gap_num'], grouped['mean'], yerr=grouped['std'],
                        fmt=style, capsize=CAPSIZE, color=color, label=arm_label)
            drawn.extend(grouped['mean'] + grouped['std'].fillna(0.0))
            drawn.extend(grouped['mean'] - grouped['std'].fillna(0.0))
            print(f"[report]   {panel_name} {arm:<4s} ntetO {motif_num}: "
                  f"{len(grouped)} gaps, mean in "
                  f"[{grouped['mean'].min():+.3f} .. {grouped['mean'].max():+.3f}], "
                  f"widest SD {np.nanmax(grouped['std'].values):.3f}")

        ax.set_ylabel(f'log2({panel_name} mean + {LOG_OFFSET_LABEL})' if i == 0 else '',
                      fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlabel(X_LABEL_GAP, fontsize=AXIS_LABEL_FONTSIZE)
        # the background grid is off (the original had it on in this figure only)
        if legend_handles is None:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend_handles = (handles, labels)
        if row_idx == 0:
            ax.set_title(f'ntetO = {motif_num}', fontsize=TITLE_FONTSIZE, fontweight='bold')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=panel_bottom(drawn))

if legend_handles is not None:
    handles, labels = legend_handles
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=LEGEND_ANCHOR,
               ncol=len(labels), fontsize=LEGEND_FONTSIZE)

for fmt in ('png', 'svg', 'pdf'):
    plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM_DISTRIBUTION}.{fmt}'), format=fmt, dpi=DPI,
                bbox_inches='tight')
plt.close()


# --- figure 2: the Fourier transform of the same curves ---------------------------------------


def compute_fft(df_tmp, n, mode, interpolate=True):
    '''The original's Fourier recipe: average the designs of one tetO count over their gap,
    put the averages on a uniform gap grid, and transform that. The gap average is taken
    after the log2, so it is the mean of the log2 levels.'''
    df_tmp = df_tmp[df_tmp['names'].str.startswith(f'spacer_{n}tetO')].copy()
    if df_tmp.empty:
        print(f"[report]   [note] no design name starts with spacer_{n}tetO")
        return None, None

    df_tmp['gap'] = df_tmp['names'].apply(extract_gap)
    df_tmp = df_tmp.dropna(subset=['gap', mode])
    if df_tmp.empty:
        print(f"[report]   [note] spacer_{n}tetO has no design with both a gap and a "
              f"drawable {mode}")
        return None, None

    df_grouped = df_tmp.groupby('gap')[mode].mean().reset_index()
    gaps_unique = df_grouped['gap'].values
    expressions_unique = df_grouped[mode].values

    if interpolate:
        gap_uniform = np.arange(gaps_unique.min(), gaps_unique.max() + 1)
        interp_func = interp1d(gaps_unique, expressions_unique, kind='linear',
                               fill_value='extrapolate')
        expressions_uniform = interp_func(gap_uniform)
    else:
        gap_uniform = gaps_unique
        expressions_uniform = expressions_unique

    fft_values = np.fft.fft(expressions_uniform)
    fft_freqs = np.fft.fftfreq(len(expressions_uniform), d=1)
    pos_mask = fft_freqs > 0
    print(f"[report]   {mode} spacer_{n}tetO: {len(df_tmp)} designs over "
          f"{len(gaps_unique)} gaps -> {len(expressions_uniform)} points from "
          f"{gaps_unique.min():.0f} to {gaps_unique.max():.0f}, {int(pos_mask.sum())} "
          f"positive frequencies")
    return fft_freqs[pos_mask], np.abs(fft_values[pos_mask])


def stack_peak_labels(ax, entries, curves, figure_height, bottom):
    '''Write each peak's period above its own peak, on a level of its own, and stop the
    dashed guide of every peak short of the labels.

    The original wrote every label at peak + 0.03 * panel maximum and drew the guide of a
    peak the full height of the panel, so the guides ran straight through the digits, and
    the labels of peaks a few gap units apart printed on top of each other in the narrow
    panels. Three rules fix that here, and all three are read off the panel's own size and
    the label font, so they hold in every column of the [8, 8, 8, 6, 4, 3] grid:

      * a label takes the lowest level that clears the curves drawn under it and every label
        already placed. The clearing is in height and not just in level: a label sits above
        its own peak, so a low level over a tall peak reaches exactly where a high level over
        a short peak does, which is how two labels land on each other on different levels.
      * a guide stops below every label whose box it would pass under, its own included.
      * the amplitude axis grows until it holds the levels that were used.

    Level 0 sits one level above the label's own peak, so a label still marks its peak.

    entries is a list of (frequency, magnitude, text, color); curves is a list of the
    (frequency, magnitude) pairs actually drawn in this panel. The new top of the axis is
    returned, or None if there was nothing to label.

    '''
    if not entries:
        return None

    x_left, x_right = ax.get_xlim()
    position = ax.get_position()
    width_inch = position.width * ax.figure.get_size_inches()[0]
    height_inch = position.height * figure_height
    if x_right <= x_left or width_inch <= 0 or height_inch <= 0:
        return None

    # one label, and one level between labels, as a fraction of the amplitude range
    label_fraction = (PEAK_FONTSIZE / 72.0) / height_inch
    level_fraction = PEAK_LANE_ASCENT * label_fraction
    x_gap = PEAK_LANE_X_GAP * (x_right - x_left)
    ordered = sorted(entries, key=lambda entry: entry[0])

    def box(freq, text):
        half = 0.5 * len(text) * PEAK_LANE_CHAR_EM * PEAK_FONTSIZE / 72.0 / width_inch \
            * (x_right - x_left)
        return freq - half, freq + half, half

    def curve_reach(x0, x1):
        '''the highest the curves drawn in this panel get between x0 and x1'''
        reach = None
        for freq, magnitude in curves:
            inside = magnitude[(freq >= x0) & (freq <= x1)]
            if inside.size:
                reach = inside.max() if reach is None else max(reach, inside.max())
        return reach

    def layout(span):
        level_height = level_fraction * span
        clearance = PEAK_LABEL_CLEARANCE * label_fraction * span
        # the marker reaches a little past its point, in both directions
        marker_x = MARKER_RADIUS_INCH / width_inch * (x_right - x_left)
        marker_y = MARKER_RADIUS_INCH / height_inch * span
        laid = []
        for freq, magnitude, text, color in ordered:
            box_left, box_right, half = box(freq, text)
            reach = curve_reach(box_left - marker_x, box_right + marker_x)
            if reach is not None:
                reach += marker_y
            level = 0
            while True:
                y = magnitude + (level + 1) * level_height
                on_curve = reach is not None and y < reach + clearance
                on_label = any(abs(freq - other[0]) < half + other[1] + x_gap
                               and abs(y - other[2]) < level_height
                               for other in laid)
                if not on_curve and not on_label:
                    break
                level += 1
            laid.append((freq, half, y, text, color))
        return laid

    # the room the labels need grows with the range they are spread over, so lay them out and
    # widen the axis until the two agree; a couple of passes settle it
    top = ax.get_ylim()[1]
    laid = []
    for _ in range(PEAK_LANE_PASSES):
        span = top - bottom
        laid = layout(span)
        needed = max(entry[2] for entry in laid) + PEAK_LANE_HEADROOM * label_fraction * span
        if needed <= top:
            break
        top = needed
    ax.set_ylim(bottom=bottom, top=top)
    label_height = label_fraction * (top - bottom)

    for freq, half, y, text, color in laid:
        ax.text(freq, y, text, fontsize=PEAK_FONTSIZE, color=color, ha='center',
                va='bottom')
    for freq, magnitude, text, color in entries:
        cutoff = min((other[2] for other in laid
                      if other[0] - other[1] <= freq <= other[0] + other[1]), default=None)
        ymax = 1.0 if cutoff is None else \
            min(max((cutoff - PEAK_LINE_TO_LABEL_GAP * label_height - bottom)
                    / (top - bottom), 0.0), 1.0)
        ax.axvline(freq, ymax=ymax, color=color, linestyle='--', alpha=PEAK_LINE_ALPHA)
    return top


# the levels the second grid transforms, as log2 columns of the merged frame
FFT_MODES = {}
for level in ('induced', 'uninduced'):
    for arm in ('FUS', 'tetR'):
        source = f'{level}_mpra_means_{arm}'
        target = f'log2_{level}_{arm}'
        values = np.asarray(df_merged[source], dtype=float)
        drawable = np.isfinite(values) & (values > -LOG_OFFSET)
        df_merged[target] = np.where(drawable, np.log2(np.where(drawable, values, 1.0) + LOG_OFFSET),
                                     np.nan)
        dropped = int((~drawable).sum())
        if dropped:
            print(f"[report] [note] {source}: {dropped} designs have no "
                  f"log2(value + {LOG_OFFSET_LABEL}) and are not transformed")
        FFT_MODES[(level, arm)] = target

FFT_PANELS = [(0, 'induced', f'EXPR FFT amplitude of log2(v + {LOG_OFFSET_LABEL})'),
              (1, 'uninduced', f'LEAK FFT amplitude of log2(v + {LOG_OFFSET_LABEL})')]

fig = plt.figure(figsize=FOURIER_FIGSIZE)
gs = GridSpec(2, len(TETO_RANGE), height_ratios=GRID_HEIGHT_RATIOS, width_ratios=COL_WIDTHS,
              hspace=GRID_HSPACE, wspace=GRID_WSPACE)

legend_handles = None
for i, ntetO in enumerate(TETO_RANGE):
    for row_idx, level, ylabel in FFT_PANELS:
        ax = fig.add_subplot(gs[row_idx, i])
        entries = []
        curves = []
        for arm, style, color, arm_label in (('FUS', FUS_STYLE, FUS_COLOR, FUS_LABEL),
                                             ('tetR', TETR_STYLE, TETR_COLOR, TETR_LABEL)):
            fft_freqs, fft_magnitude = compute_fft(df_merged, n=ntetO,
                                                   mode=FFT_MODES[(level, arm)])
            if fft_freqs is None:
                continue
            ax.plot(fft_freqs, fft_magnitude, style, color=color, label=arm_label)
            curves.append((fft_freqs, fft_magnitude))

            periods = []
            peaks, _ = find_peaks(fft_magnitude)
            for idx in peaks:
                freq = fft_freqs[idx]
                period = 1 / freq
                periods.append(f'{period:.1f}')
                entries.append((freq, fft_magnitude[idx], f'{period:.1f}', color))
            print(f"[report]   {level} {arm:<4s} ntetO {ntetO}: peaks at gap periods "
                  f"{', '.join(periods) if periods else '(none)'}")

        if row_idx == 0:
            ax.set_title(f'ntetO = {ntetO}', fontsize=TITLE_FONTSIZE, fontweight='bold')
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_ylabel('')
        if row_idx == 1:
            ax.set_xlabel(X_LABEL_FREQ, fontsize=AXIS_LABEL_FONTSIZE)
        if legend_handles is None:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend_handles = (handles, labels)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=FOURIER_YLIM_BOTTOM)
        stack_peak_labels(ax, entries, curves, FOURIER_FIGSIZE[1], FOURIER_YLIM_BOTTOM)

if legend_handles is not None:
    handles, labels = legend_handles
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=LEGEND_ANCHOR,
               ncol=len(labels), fontsize=LEGEND_FONTSIZE)

for fmt in ('png', 'svg', 'pdf'):
    plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM_FOURIER}.{fmt}'), format=fmt, dpi=DPI,
                bbox_inches='tight')
plt.close()
