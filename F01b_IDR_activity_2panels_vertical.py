import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from scipy import stats

# Project layout: data in ./data, figures in ./figs (checkpoints, where used, in ./checks)
DATA_DIR = './data'
FIG_DIR = './figs'

INPUT_CSV = os.path.join(DATA_DIR, 'F1_IDR_activity.csv')
FIG_STEM = 'F01b_IDR_activity_vertical'
FET_IDRS = ['FUS', 'EWSR1', 'TAF15']
CONTROL_IDR = 'control'
DPI = 600

# --- NATURE PRESS GLOBAL FONT CONFIGURATION ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']  # Preferred journal fonts
plt.rcParams.update({
    'font.size': 5,          # Standard micro-label size
    'axes.labelsize': 5,     # Axis titles (6 pt)
    'axes.titlesize': 5,     # Panel titles (6 pt)
    'xtick.labelsize': 5,    # X tick text
    'ytick.labelsize': 5,    # Y tick text
    'lines.linewidth': 0.25  # Cleaner line weight for small figures
})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

os.makedirs(FIG_DIR, exist_ok=True)

# Load data from CSV file
df = pd.read_csv(INPUT_CSV)

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
if sample_names[pc_idx] != CONTROL_IDR:
    raise ValueError(
        f"Row 0 of {INPUT_CSV} is {sample_names[pc_idx]!r}, expected the control {CONTROL_IDR!r}; "
        "the one-sample t-tests below are run against the first row."
    )
mean_expression_pc = float(exp_means[pc_idx])
mean_leak_pc = float(leak_means[pc_idx])

# Positions
n_samples = len(sample_names)
x = np.arange(n_samples)

# --- EXACT CANVAS DIMENSIONS (75mm x 59mm) ---

width_inch = 75 / 25.4
height_inch = 59 / 25.4
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width_inch, height_inch), sharex=True)

# TOP PANEL: Induction (dark blue bars)
bars1 = ax1.bar(x, exp_means, yerr=exp_stds, capsize=2, color='#012f48', width=0.5,
                alpha=0.8, ecolor='black', error_kw={'linewidth': 0.25, 'capthick': 0.25})
ax1.tick_params(axis='y', labelcolor='#012f48', pad=1)
ax1.set_title('Induced Activity', color='#012f48', fontweight='bold', pad=3)
ax1.set_ylim(0, 25000)
formatter1 = ScalarFormatter(useMathText=True) # Enables the pretty 10^x format
formatter1.set_scientific(True)
formatter1.set_powerlimits((3, 3))            # Forces exactly 10^3
ax1.yaxis.set_major_formatter(formatter1)
# Small adjustment: move the "10^3" multiplier text so it doesn't overlap titles
ax1.yaxis.get_offset_text().set_fontsize(5)


# BOTTOM PANEL: Leakiness (light blue bars)
bars2 = ax2.bar(x, leak_means, yerr=leak_stds, capsize=2, color='#669aba', width=0.5,
                alpha=0.8, ecolor='black', error_kw={'linewidth': 0.25, 'capthick': 0.25})
ax2.tick_params(axis='y', labelcolor='#669aba', pad=0.5)
ax2.tick_params(axis='x', pad=1)
ax2.set_title('Leakiness', color='#669aba', fontweight='bold', pad=3)
ax2.set_xlabel('IDR', labelpad=1.0)  # Only bottom panel has x-label
ax2.set_ylim(0, 600)
formatter2 = ScalarFormatter(useMathText=True) # Enables the pretty 10^x format
formatter2.set_scientific(True)
formatter2.set_powerlimits((1, 1))            # Forces exactly 10^1
ax2.yaxis.set_major_formatter(formatter2)
# Small adjustment: move the "10^1" multiplier text so it doesn't overlap titles
ax2.yaxis.get_offset_text().set_fontsize(5)

# SINGLE SHARED Y-AXIS LABEL (Centered vertically across both panels)
fig.supylabel('Fluorescence intensity a.u.', fontweight='bold', fontsize=5, x=0.08)

# T-TESTS
pvals_exp = []
pvals_leak = []
for i in range(1, n_samples):
    reps_exp = df[exp_cols].iloc[i].astype(np.float64).values
    reps_leak = df[leak_cols].iloc[i].astype(np.float64).values
    _, p_exp = stats.ttest_1samp(reps_exp, mean_expression_pc)
    _, p_leak = stats.ttest_1samp(reps_leak, mean_leak_pc)
    pvals_exp.append(p_exp)
    pvals_leak.append(p_leak)


# Significance function
def add_significance(ax, pvals, means, stds):
    for sample_idx, p in zip(range(1, n_samples), pvals):
        if p < 0.001: stars = '***'
        elif p < 0.01: stars = '**'
        elif p < 0.05: stars = '*'
        else: stars = 'ns'
        x_pos = x[sample_idx]
        y_pos = means[sample_idx] + stds[sample_idx] * 1.2
        ax.text(x_pos, y_pos, stars, ha='center', va='bottom',
                fontsize=5)

add_significance(ax1, pvals_exp, exp_means, exp_stds)
add_significance(ax2, pvals_leak, leak_means, leak_stds)

# Shared formatting
#fig.suptitle('Expression and Leakiness vs Positive Control (1-sample t-test)',
#             fontsize=16, fontweight='bold', y=0.98)

ax2.set_xticks(x)
ax2.set_xticklabels(sample_names, rotation=90, ha='center')

# --- GLOBAL TICK ADJUSTMENTS (Length and Top Panel Cleanup) ---
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', which='major', length=1, width=0.25)

    for spine in ax.spines.values():
        spine.set_linewidth(0.25)

# Completely remove the x-axis ticks from the top panel
ax1.tick_params(axis='x', which='both', bottom=False, top=False)

fig.subplots_adjust(hspace=0.22, bottom=0.18, top=0.95, left=0.18, right=0.95)
#for ax in [ax1, ax2]:
#    ax.grid(True, alpha=0.3, axis='y')

#plt.tight_layout()

# Save as vector formats for direct Adobe Illustrator scaling/editing
plt.savefig(os.path.join(FIG_DIR, FIG_STEM + '.pdf'), format='pdf', transparent=True)
plt.savefig(os.path.join(FIG_DIR, FIG_STEM + '.svg'), format='svg', transparent=True)
plt.savefig(os.path.join(FIG_DIR, FIG_STEM + '.png'), format='png', transparent=True, dpi=DPI)
plt.close()

print("Expression p-values:", [f"{p:.3e}" for p in pvals_exp])
print("Leakiness p-values:", [f"{p:.3e}" for p in pvals_leak])


# --- REPORT: FET family vs control, the values quoted in the Fig. 1B (right) sentence ---

def fmt(value):
    return f"{value:.2f}"

print("[report] Per IDR:")
for i, name in enumerate(sample_names):
    print(f"[report]   {name}: Induction {fmt(exp_means[i])} +/- {fmt(exp_stds[i])}, "
          f"Leakiness {fmt(leak_means[i])} +/- {fmt(leak_stds[i])}")
