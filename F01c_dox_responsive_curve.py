import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Project layout: data in ./data, figures in ./figs.
DATA_DIR = './data'
FIG_DIR = './figs'
INPUT_XLSX = os.path.join(DATA_DIR, 'F1_dox responsive curve plot.xlsx')
FIG_STEM = 'F01c_dox_responsive_curve'
DPI = 600

# --- NATURE PRESS GLOBAL FONT & STYLING CONFIGURATION (as in the reference) ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['svg.fonttype'] = 'none'  # Keeps SVG text fully editable
plt.rcParams.update({
    'font.size': 5,
    'axes.labelsize': 5,
    'axes.titlesize': 5,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'lines.linewidth': 0.25,
})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

os.makedirs(FIG_DIR, exist_ok=True)

if not os.path.exists(INPUT_XLSX):
    raise FileNotFoundError(
        f"{INPUT_XLSX} not found. This dose-response table is not derivable from "
        "the other project tables; copy the reference workbook "
        "20251103-dox responsive curve plot.xlsx into ./data/ under the name "
        "'F1_dox responsive curve plot.xlsx'."
    )

# === 1. Read Excel ===
data = pd.read_excel(INPUT_XLSX, header=None)

if data.shape[0] < 7 or data.shape[1] < 2:
    raise ValueError(
        f"{INPUT_XLSX} has shape {data.shape}; expected row 0 with the Dox "
        "concentrations from column 1 onward and rows 1-6 with the six replicates."
    )

labels = data.iloc[:, 0].astype(str).str.strip().tolist()
if not all(label.startswith('Tet-on') for label in labels[1:4]):
    raise ValueError(
        f"rows 1-3 of {INPUT_XLSX} are {labels[1:4]}, expected the three "
        "'Tet-on-rep' rows (the rtTA arm); the reference slices these rows by position."
    )
if not all(label.startswith('FUS-Tet-on') for label in labels[4:7]):
    raise ValueError(
        f"rows 4-6 of {INPUT_XLSX} are {labels[4:7]}, expected the three "
        "'FUS-Tet-on-rep' rows (the FUS-rtTA arm); the reference slices these rows by position."
    )

# Drop first column (labels)
data_only = data.iloc[:, 1:]

# Force numeric values, so a text cell fails loudly instead of entering the fit
data_numeric = data_only.apply(pd.to_numeric, errors='coerce').astype(float)
if data_numeric.isna().any().any():
    bad = np.argwhere(data_numeric.isna().values)
    raise ValueError(
        f"{INPUT_XLSX} has non-numeric or empty cells at (row, value column) "
        f"{[tuple(index) for index in bad]}; the reference converted the block to float."
    )

# === 2. Extract data ===
concs = data_numeric.iloc[0].values.astype(float)

# Rows 1-3: sample 1 replicates (rtTA)
sample1_reps = data_numeric.iloc[1:4].values.astype(float)

# Rows 4-6: sample 2 replicates (FUS-rtTA)
sample2_reps = data_numeric.iloc[4:7].values.astype(float)

# Means for fitting
s1_mean = sample1_reps.mean(axis=0)
s2_mean = sample2_reps.mean(axis=0)

# Sort by concentration
sort_idx = np.argsort(concs)
concs_sorted = concs[sort_idx]
s1_mean_sorted = s1_mean[sort_idx]
s2_mean_sorted = s2_mean[sort_idx]
s1_reps_sorted = sample1_reps[:, sort_idx]
s2_reps_sorted = sample2_reps[:, sort_idx]


# === 3. Hill function and fits ===
def hill(x, bottom, top, EC50, nH):
    return bottom + (top - bottom) * (x**nH) / (EC50**nH + x**nH)


# Use only positive concentrations for fitting and plotting on log scale
positive_mask = concs_sorted > 0
x1 = concs_sorted[positive_mask]
y1 = s1_mean_sorted[positive_mask]
x2 = concs_sorted[positive_mask]
y2 = s2_mean_sorted[positive_mask]

s1_reps_pos = s1_reps_sorted[:, positive_mask]
s2_reps_pos = s2_reps_sorted[:, positive_mask]

# a four-parameter Hill fit needs at least four distinct positive concentrations
if positive_mask.sum() < 4:
    raise ValueError(
        f"only {positive_mask.sum()} positive Dox concentrations in {INPUT_XLSX}; "
        "the four-parameter Hill fit needs at least four."
    )
if not (np.isfinite(y1).all() and np.isfinite(y2).all() and (y1 > 0).all() and (y2 > 0).all()):
    raise ValueError(
        f"{INPUT_XLSX} has non-positive or non-finite mean fluorescence at a positive "
        "Dox concentration; the log y axis and the Hill fit cannot be drawn."
    )

p0_1 = [y1.min(), y1.max(), np.median(x1), 1.0]
p0_2 = [y2.min(), y2.max(), np.median(x2), 1.0]

popt1, _ = curve_fit(hill, x1, y1, p0=p0_1, maxfev=10000)
popt2, _ = curve_fit(hill, x2, y2, p0=p0_2, maxfev=10000)

bottom1, top1, EC50_1, nH1 = popt1
bottom2, top2, EC50_2, nH2 = popt2

# Dense x for smooth curves (positive range only)
x_fit = np.logspace(np.log10(x1.min()), np.log10(x1.max()), 200)
y_fit1 = hill(x_fit, *popt1)
y_fit2 = hill(x_fit, *popt2)

# === 4. Plot replicates + fitted curves (Exact Nature Canvas Size) ===
width_inch = 55 / 25.4
height_inch = 45 / 25.4
fig, ax = plt.subplots(figsize=(width_inch, height_inch))

# Sample 1 replicates (rtTA): circles
for i in range(s1_reps_pos.shape[0]):
    ax.scatter(
        x1,
        s1_reps_pos[i],
        marker='o',
        color='black',
        s=6,  # scaled marker size for small canvas
        alpha=1,
        label='rtTA' if i == 0 else None,
    )

# Sample 2 replicates (FUS-rtTA): crosses
for i in range(s2_reps_pos.shape[0]):
    ax.scatter(
        x2,
        s2_reps_pos[i],
        marker='x',
        color='#7455a3',
        s=10,
        alpha=1,
        label='FUS-rtTA' if i == 0 else None,
    )

# Fitted Hill curves
ax.plot(
    x_fit,
    y_fit1,
    color='black',
    linewidth=0.75,
    label='rtTA Hill fit ',
)
ax.plot(
    x_fit,
    y_fit2,
    color='#7455a3',
    linewidth=0.75,
    label='FUS-rtTA Hill fit ',
)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Dox concentration (ng/mL)', fontweight='bold', labelpad=1.0)
ax.set_ylabel('Fluorescence Intensity a.u.', fontweight='bold', labelpad=1.0)
ax.set_title('Dox Responsive Curve', fontweight='bold', pad=3)

# Legend, Ticks, and Spines formatting matching F1_IDR_activity
ax.legend(
    frameon=False,
    loc='upper left',
    fontsize=4,
    borderaxespad=0.3,
    handletextpad=0.3,
)
ax.tick_params(axis='both', which='both', length=1, width=0.25, pad=1)
for spine in ax.spines.values():
    spine.set_linewidth(0.25)

fig.subplots_adjust(bottom=0.18, top=0.95, left=0.18, right=0.95)

# --- SAVE (pdf and svg as in the reference, plus png) --------------------------

plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM}.pdf'), format='pdf', transparent=True)
plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM}.svg'), format='svg', transparent=False)
plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM}.png'), format='png', transparent=True,
            dpi=DPI)
plt.close()

print(f"[report] {positive_mask.sum()} positive Dox concentrations per construct "
      f"enter the fit and plot; the 0 ng/mL column is dropped (log x axis).")
print('Sample 1 params:', popt1)
print('Sample 2 params:', popt2)
print(f'TetR Hill fit (EC50={EC50_1:.2g}, nH={nH1:.2g})')
print(f'FUS-TetR Hill fit (EC50={EC50_2:.2g}, nH={nH2:.2g})')
