import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from scipy import stats

# Project layout: data in ./data, figures in ./figs.
DATA_DIR = './data'
FIG_DIR = './figs'
INPUT_CSV = os.path.join(DATA_DIR, 'F4_double_mutant_activity.csv')
FIG_STEM = 'F01f_double_mutant_activity_vertical'
CONTROL_ALIASES = ('WT', 'Baseline TRE')
DPI = 600
# Reference value is 0.18; raised so the rotated x tick labels and x label are not
# clipped at the bottom of the 35 x 59 mm canvas (see the module docstring).
BOTTOM_MARGIN = 0.28

# --- NATURE PRESS GLOBAL FONT CONFIGURATION (as in the reference) -------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']  # Preferred journal fonts
plt.rcParams.update({
    'font.size': 5,          # Standard micro-label size
    'axes.labelsize': 5,     # Axis titles (6 pt)
    'axes.titlesize': 5,     # Panel titles (6 pt)
    'xtick.labelsize': 4.5,  # X tick text
    'ytick.labelsize': 5,    # Y tick text
    'lines.linewidth': 0.25  # Cleaner line weight for small figures
})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

os.makedirs(FIG_DIR, exist_ok=True)

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"{INPUT_CSV} not found. This fluorescence table is not derivable from the "
        "MPRA double-mutant table or from the Figure 6 reporter workbooks; copy "
        "F4_double_mutant_activity.csv into ./data/."
    )

# Load data from CSV file
df = pd.read_csv(INPUT_CSV)

required_cols = ['sample', 'exp_rep1', 'exp_rep2', 'exp_rep3',
                 'leak_rep1', 'leak_rep2', 'leak_rep3']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(
        f"{INPUT_CSV} is missing the column(s) {missing_cols}; expected "
        f"{required_cols}."
    )

sample_names = df['sample'].tolist()
exp_cols = ['exp_rep1', 'exp_rep2', 'exp_rep3']
leak_cols = ['leak_rep1', 'leak_rep2', 'leak_rep3']

# Force numeric columns
df[exp_cols + leak_cols] = df[exp_cols + leak_cols].apply(pd.to_numeric, errors='coerce').astype(float)

# Compute means/stds
exp_means = df[exp_cols].mean(axis=1).tolist()
exp_stds = df[exp_cols].std(ddof=1, axis=1).tolist()
leak_means = df[leak_cols].mean(axis=1).tolist()
leak_stds = df[leak_cols].std(ddof=1, axis=1).tolist()

# PC (first row)
pc_idx = 0
if sample_names[pc_idx] not in CONTROL_ALIASES:
    raise ValueError(
        f"Row 0 of {INPUT_CSV} is {sample_names[pc_idx]!r}, expected the control "
        f"{CONTROL_ALIASES[0]!r} (or its alias {CONTROL_ALIASES[1]!r}); the "
        "one-sample t-tests below are run against the first row."
    )
mean_expression_pc = float(exp_means[pc_idx])
mean_leak_pc = float(leak_means[pc_idx])

# Positions
n_samples = len(sample_names)
x = np.arange(n_samples)

# --- EXACT CANVAS DIMENSIONS (Vertical Stack) ---
width_inch = 35 / 25.4
height_inch = 59 / 25.4
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width_inch, height_inch), sharex=True)

# TOP PANEL: Expression (green bars)
bars1 = ax1.bar(x, exp_means, yerr=exp_stds, capsize=2, color='#012f48', width=0.5,
                alpha=0.8, ecolor='black', error_kw={'linewidth': 0.25, 'capthick': 0.25})
ax1.tick_params(axis='y', labelcolor='#012f48', pad=1)
ax1.set_title('Induced Activity', color='#012f48', fontweight='bold', pad=3)
ax1.set_ylim(0, 12000)
formatter1 = ScalarFormatter(useMathText=True) # Enables the pretty 10^x format
formatter1.set_scientific(True)
formatter1.set_powerlimits((3, 3))            # Forces exactly 10^3
ax1.yaxis.set_major_formatter(formatter1)
ax1.yaxis.get_offset_text().set_fontsize(5)

# BOTTOM PANEL: Leakiness (grey bars)
bars2 = ax2.bar(x, leak_means, yerr=leak_stds, capsize=2, color='#669aba', width=0.5,
                alpha=0.8, ecolor='black', error_kw={'linewidth': 0.25, 'capthick': 0.25})
ax2.tick_params(axis='y', labelcolor='#669aba', pad=0.5)
ax2.tick_params(axis='x', pad=1)
ax2.set_title('Leakiness', color='#669aba', fontweight='bold', pad=3)
ax2.set_xlabel('Double Mutant Variant', labelpad=1.5)  # Only bottom panel has x-label
ax2.set_ylim(0, 250)
formatter2 = ScalarFormatter(useMathText=True)
formatter2.set_scientific(True)
formatter2.set_powerlimits((1, 1))
ax2.yaxis.set_major_formatter(formatter2)
ax2.yaxis.get_offset_text().set_fontsize(5)

# SINGLE SHARED Y-AXIS LABEL (Centered vertically across both panels)
fig.supylabel('Fluorescence intensity a.u.', fontweight='bold', fontsize=5, x=0.01)


def one_sample_p(reps, mu):
    """Reference behaviour: a NaN replicate makes the t-test p-value NaN."""
    reps = np.asarray(reps, dtype=np.float64)
    if np.isnan(reps).any():
        return np.nan
    return float(stats.ttest_1samp(reps, mu).pvalue)


# T-TESTS
pvals_exp = []
pvals_leak = []
for i in range(1, n_samples):
    reps_exp = df[exp_cols].iloc[i].astype(np.float64).values
    reps_leak = df[leak_cols].iloc[i].astype(np.float64).values
    pvals_exp.append(one_sample_p(reps_exp, mean_expression_pc))
    pvals_leak.append(one_sample_p(reps_leak, mean_leak_pc))


# Significance function
def add_significance(ax, pvals, means, stds):
    for sample_idx, p in zip(range(1, n_samples), pvals):
        if p < 0.001: stars = '***'
        elif p < 0.01: stars = '**'
        elif p < 0.05: stars = '*'
        else: stars = 'ns'
        x_pos = x[sample_idx]
        y_pos = means[sample_idx] + stds[sample_idx] * 1.2
        ax.text(x_pos, y_pos, stars, ha='center', va='bottom', fontsize=5)

add_significance(ax1, pvals_exp, exp_means, exp_stds)
add_significance(ax2, pvals_leak, leak_means, leak_stds)

ax2.set_xticks(x)
ax2.set_xticklabels(sample_names, rotation=90, ha='center')

# --- GLOBAL TICK ADJUSTMENTS ---
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', which='major', length=1, width=0.25)
    for spine in ax.spines.values():
        spine.set_linewidth(0.25)

# Completely remove the x-axis ticks from top panel
ax1.tick_params(axis='x', which='both', bottom=False, top=False)

fig.subplots_adjust(hspace=0.22, bottom=BOTTOM_MARGIN, top=0.95, left=0.18, right=0.95)

# Save figure
plt.savefig(os.path.join(FIG_DIR, FIG_STEM + '.pdf'), format='pdf', transparent=True)
plt.savefig(os.path.join(FIG_DIR, FIG_STEM + '.svg'), format='svg', transparent=True)
plt.savefig(os.path.join(FIG_DIR, FIG_STEM + '.png'), format='png', transparent=True, dpi=DPI)
plt.close()

print("Expression p-values:", [f"{p:.3e}" for p in pvals_exp])
print("Leakiness p-values:", [f"{p:.3e}" for p in pvals_leak])


# --- REPORT: the values behind the bars and stars -----------------------------
def _fmt(value):
    return f"{value:.2f}"


def _stars(p):
    if np.isnan(p):
        return 'ns (p=nan)'
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns'


print(f"[report] Control (row 0, {sample_names[pc_idx]}): "
      f"Induction {_fmt(exp_means[pc_idx])} +/- {_fmt(exp_stds[pc_idx])}, "
      f"Leakiness {_fmt(leak_means[pc_idx])} +/- {_fmt(leak_stds[pc_idx])}")
print("[report] Per sample (mean +/- SD, ddof=1; p from one-sample t-test vs row 0):")
for i, name in enumerate(sample_names):
    if i == pc_idx:
        continue
    print(f"[report]   {name}: "
          f"Induction {_fmt(exp_means[i])} +/- {_fmt(exp_stds[i])} ({_stars(pvals_exp[i-1])}), "
          f"Leakiness {_fmt(leak_means[i])} +/- {_fmt(leak_stds[i])} ({_stars(pvals_leak[i-1])})")

missing_leak = df[leak_cols].isna().sum(axis=1)
for i, n_missing in enumerate(missing_leak.tolist()):
    if n_missing:
        print(f"[report] Note: {sample_names[i]} has {n_missing} missing leak replicate(s); "
              "means/SDs use the available replicates and the reference's t-test gives "
              "p=nan, which is drawn as 'ns'.")
