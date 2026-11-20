import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Project layout: data in ./data, figures in ./figs.
DATA_DIR = './data'
FIG_DIR = './figs'
INPUT_CSV = os.path.join(DATA_DIR, 'toxic_gene_expression_raw_data_V7.csv')
FIG_STEM = 'F02c_Cell_Viability_Plot_V7'
DPI = 600

# ==========================================
# 1. ENVIRONMENT & FONTS CONFIGURATION
# ==========================================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update(
    {
        "font.size": 5,
        "axes.labelsize": 5,
        "axes.titlesize": 5,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "lines.linewidth": 0.25,
    }
)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

os.makedirs(FIG_DIR, exist_ok=True)

# ==========================================
# 2. FILE LOADING & COLUMN-INDEX MAPPING
# ==========================================
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"{INPUT_CSV} not found. This is the raw viability table for the figure; "
        "place toxic_gene_expression_raw_data_V7.csv in ./data/."
    )

raw_df = pd.read_csv(INPUT_CSV, header=None)

column_definitions = {
    1: {"Sample": "S1", "Dox": "+"},
    2: {"Sample": "S1", "Dox": "-"},
    3: {"Sample": "S3", "Dox": "+"},
    4: {"Sample": "S3", "Dox": "-"},
    5: {"Sample": "S4", "Dox": "+"},
    6: {"Sample": "S4", "Dox": "-"},
    7: {"Sample": "S6", "Dox": "+"},
    8: {"Sample": "S6", "Dox": "-"},
    9: {"Sample": "S7", "Dox": "+"},
    10: {"Sample": "S7", "Dox": "-"},
    11: {"Sample": "Mock", "Dox": "+"},
    12: {"Sample": "Mock", "Dox": "-"},
}

# Row 0 = sample, 1 = cis-RE, 2 = trans-RE, 3 = GCV, 4 = Dox, 5:14 = rep1..rep9.
FIRST_REP_ROW = 5
LAST_REP_ROW = 14
if raw_df.shape[0] < LAST_REP_ROW:
    raise ValueError(
        f"{INPUT_CSV} has {raw_df.shape[0]} rows; the reference layout needs at "
        f"least {LAST_REP_ROW} (header, cis-RE, trans-RE, GCV, Dox, rep1..rep9)."
    )
if raw_df.shape[1] <= max(column_definitions):
    raise ValueError(
        f"{INPUT_CSV} has {raw_df.shape[1]} columns; the reference column map "
        f"needs columns 0..{max(column_definitions)}."
    )

# The map above is hard-coded, so check it against the file rather than trusting it.
for row_idx, expected_label in ((1, 'cis-RE'), (2, 'trans-RE'), (4, 'Dox')):
    actual_label = str(raw_df.iloc[row_idx, 0]).strip()
    if actual_label != expected_label:
        raise ValueError(
            f"{INPUT_CSV}: row {row_idx} starts with {actual_label!r}, expected "
            f"{expected_label!r}; the row layout is not the reference layout."
        )

sample_row = raw_df.iloc[0].values
dox_row = raw_df.iloc[4].values
cis_row = raw_df.iloc[1].values
trans_row = raw_df.iloc[2].values

for col_idx, meta in column_definitions.items():
    file_sample = str(sample_row[col_idx]).strip()
    file_dox = str(dox_row[col_idx]).strip()
    if file_sample != meta["Sample"] or file_dox != meta["Dox"]:
        raise ValueError(
            f"{INPUT_CSV}, column {col_idx}: found Sample={file_sample!r}, "
            f"Dox={file_dox!r}; the hard-coded map expects "
            f"Sample={meta['Sample']!r}, Dox={meta['Dox']!r}."
        )
    if pd.isna(cis_row[col_idx]) or pd.isna(trans_row[col_idx]):
        raise ValueError(
            f"{INPUT_CSV}, column {col_idx}: missing cis-RE or trans-RE value."
        )

parsed_records = []

for col_idx, meta in column_definitions.items():
    cis_val = str(cis_row[col_idx]).strip()
    trans_val = str(trans_row[col_idx]).strip()

    replicate_series = pd.to_numeric(
        raw_df.iloc[FIRST_REP_ROW:LAST_REP_ROW, col_idx], errors="coerce"
    ).reset_index(drop=True)

    for rep_idx, val in enumerate(replicate_series):
        if pd.notna(val):
            parsed_records.append(
                {
                    "Sample": meta["Sample"],
                    "cis-RE": cis_val,
                    "trans-RE": trans_val,
                    "Dox": meta["Dox"],
                    "Replicate": f"rep{rep_idx + 1}",
                    "Normalized_Viability": val,
                }
            )

df = pd.DataFrame(parsed_records)
if df.empty:
    raise ValueError(f"{INPUT_CSV} contains no numeric replicate values.")

# ==========================================
# 3. IQR OUTLIER REMOVAL FUNCTION
# ==========================================


def filter_iqr_outliers(data_df, value_col):
    clean_dfs = []
    grouped = data_df.groupby(["Sample", "Dox"])

    for _, group in grouped:
        vals = group[value_col].values
        if len(vals) > 0:
            q25, q75 = np.percentile(vals, 25), np.percentile(vals, 75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            filtered = group[
                (group[value_col] >= lower_bound)
                & (group[value_col] <= upper_bound)
            ]
            clean_dfs.append(filtered)

    return pd.concat(clean_dfs, ignore_index=True)


n_raw = len(df)
df_clean = filter_iqr_outliers(df, "Normalized_Viability")
n_clean = len(df_clean)

# ==========================================
# 4. GROUP STATISTICAL SUMMARY & PAIRWISE T-TESTS
# ==========================================
summary_list = []
grouped = df_clean.groupby(["Sample", "cis-RE", "trans-RE", "Dox"])

for (sample_name, cis_re, trans_re, dox_status), group in grouped:
    group_values = group["Normalized_Viability"].values
    mean_val = group_values.mean()
    sd_val = group_values.std(ddof=1) if len(group_values) > 1 else 0

    summary_list.append(
        {
            "Sample": sample_name,
            "cis-RE": cis_re,
            "trans-RE": trans_re,
            "Dox": dox_status,
            "Mean_Viability": mean_val,
            "SD": sd_val,
        }
    )

df_summary = pd.DataFrame(summary_list)
if df_summary.empty:
    raise ValueError("No groups remain after IQR filtering.")

# One row per sample and Dox condition is required by the .loc[] selection below.
n_per_sample_dox = df_summary.groupby(["Sample", "Dox"]).size()
if (n_per_sample_dox > 1).any():
    duplicated = n_per_sample_dox[n_per_sample_dox > 1].index.tolist()
    raise ValueError(
        f"{INPUT_CSV}: samples {duplicated} have more than one cis/trans group "
        "for the same Dox condition; the reference assumes one group per "
        "(Sample, Dox)."
    )


def get_p_and_stars(sample_a, sample_b, dox_cond="-"):
    vals_a = df_clean[
        (df_clean["Sample"] == sample_a) & (df_clean["Dox"] == dox_cond)
    ]["Normalized_Viability"].values
    vals_b = df_clean[
        (df_clean["Sample"] == sample_b) & (df_clean["Dox"] == dox_cond)
    ]["Normalized_Viability"].values
    _, p_val = stats.ttest_ind(vals_a, vals_b, equal_var=False)
    stars = (
        "***"
        if p_val < 0.001
        else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    )
    return p_val, stars


p_s3_s7, stars_s3_s7 = get_p_and_stars("S3", "S7", dox_cond="-")
p_s1_s7, stars_s1_s7 = get_p_and_stars("S1", "S7", dox_cond="-")

sample_order = ["S1", "S3", "S4", "S6", "S7", "Mock"]
unique_samples = [s for s in sample_order if s in df_summary["Sample"].unique()]

required_samples = ["S1", "S3", "S7"]
missing_required = [s for s in required_samples if s not in unique_samples]
if missing_required:
    raise ValueError(
        f"{INPUT_CSV}: sample(s) {missing_required} are required by the two "
        "comparison brackets but are missing from the data."
    )
if len(unique_samples) != len(sample_order):
    print(f"[report] Warning: samples {[s for s in sample_order if s not in unique_samples]} "
          "are missing; the reference would silently drop them, this script keeps "
          "the same order for the samples that are present.")

for dox_cond in ("-", "+"):
    missing_dox = [
        s for s in unique_samples
        if s not in set(df_summary.loc[df_summary["Dox"] == dox_cond, "Sample"])
    ]
    if missing_dox:
        raise ValueError(
            f"{INPUT_CSV}: sample(s) {missing_dox} have no Dox{dox_cond} group; "
            "each sample needs both Dox conditions."
        )

dox_minus_data = (
    df_summary[df_summary["Dox"] == "-"]
    .set_index("Sample")
    .loc[unique_samples]
    .reset_index()
)
dox_plus_data = (
    df_summary[df_summary["Dox"] == "+"]
    .set_index("Sample")
    .loc[unique_samples]
    .reset_index()
)

# ==========================================
# 5. STANDARDIZED GRAPHICAL RENDERING
# ==========================================
width_inch = 105 / 25.4
height_inch = 50 / 25.4

fig, ax = plt.subplots(figsize=(width_inch, height_inch))

x = np.arange(len(unique_samples))
width = 0.26
bar_offset = 0.16

bars_dox_minus = ax.bar(
    x - bar_offset,
    dox_minus_data["Mean_Viability"],
    width,
    yerr=dox_minus_data["SD"],
    label="Dox-",
    capsize=2,
    color="#669aba",
    alpha=0.8,
    edgecolor="none",
    linewidth=0.25,
    error_kw={"linewidth": 0.25, "capthick": 0.25, "ecolor": "black"},
)

bars_dox_plus = ax.bar(
    x + bar_offset,
    dox_plus_data["Mean_Viability"],
    width,
    yerr=dox_plus_data["SD"],
    label="Dox+",
    capsize=2,
    color="#012f48",
    alpha=0.8,
    edgecolor="none",
    linewidth=0.25,
    error_kw={"linewidth": 0.25, "capthick": 0.25, "ecolor": "black"},
)

# Overlay individual dots in uniform black
np.random.seed(42)
jitter = 0.04

for idx, sample in enumerate(unique_samples):
    pts_minus = df_clean[
        (df_clean["Sample"] == sample) & (df_clean["Dox"] == "-")
    ]["Normalized_Viability"].values
    x_pos_minus = (idx - bar_offset) + np.random.uniform(
        -jitter, jitter, size=len(pts_minus)
    )
    ax.scatter(
        x_pos_minus,
        pts_minus,
        color="black",
        s=4,
        zorder=4,
        alpha=0.8,
        linewidths=0,
    )

    pts_plus = df_clean[
        (df_clean["Sample"] == sample) & (df_clean["Dox"] == "+")
    ]["Normalized_Viability"].values
    x_pos_plus = (idx + bar_offset) + np.random.uniform(
        -jitter, jitter, size=len(pts_plus)
    )
    ax.scatter(
        x_pos_plus,
        pts_plus,
        color="black",
        s=4,
        zorder=4,
        alpha=0.8,
        linewidths=0,
    )

# ----------------------------------------------------
# PAIRWISE COMPARISON BRACKETS
# ----------------------------------------------------
s1_idx = unique_samples.index("S1") - bar_offset
s3_idx = unique_samples.index("S3") - bar_offset
s7_idx = unique_samples.index("S7") - bar_offset

# Find the maximum bar top + SD across all data to position brackets tightly
max_top = max(df_summary["Mean_Viability"] + df_summary["SD"])

base_y = max_top * 1.0  # Reduced from 1.12 to sit closer to the highest error bar
bracket_h = 2.0  # Absolute or smaller relative height for the bracket legs
step_y = 9.0  # Controlled vertical gap between the first and second bracket


def draw_bracket(x1, x2, y_pos, stars):
  ax.plot(
      [x1, x1, x2, x2],
      [y_pos, y_pos + bracket_h, y_pos + bracket_h, y_pos],
      lw=0.25,
      c="black",
  )
  ax.text(
      (x1 + x2) * 0.5,
      y_pos + bracket_h,
      stars,
      ha="center",
      va="bottom",
      color="black",
      fontsize=5,
  )


draw_bracket(s3_idx, s7_idx, base_y, stars_s3_s7)
draw_bracket(s1_idx, s7_idx, base_y + step_y, stars_s1_s7)

for spine in ax.spines.values():
    spine.set_linewidth(0.25)
    spine.set_color("black")

ax.set_ylabel("Relative Cell Viability (%)", fontweight="bold", labelpad=1)
ax.tick_params(axis="y", colors="black", width=0.25, length=1, pad=1)

# Set ylim comfortably above the top bracket so everything stays inside the frame
ax.set_ylim(0, base_y + step_y + 19)

ax.set_xticks(x)
ax.set_xticklabels([])
ax.set_xlabel("")

ax.legend(
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(1.0, 1.0),
    handletextpad=0.3,
    borderaxespad=0.2,
    fontsize=4,
)

plt.subplots_adjust(bottom=0.22, left=0.1, right=0.95, top=0.95)

label_x_pos = -0.55
y_offset_step = -0.06

table_rows = [
    ("Trans-RE:", dox_minus_data["trans-RE"].values),
    ("Cis-RE:", dox_minus_data["cis-RE"].values),
]

for idx, (row_title, column_values) in enumerate(table_rows):
    current_y = y_offset_step * (idx + 1)
    ax.text(
        label_x_pos,
        current_y,
        row_title,
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="center",
        fontweight="bold",
        color="black",
    )

    for col_idx, val in enumerate(column_values):
        ax.text(
            col_idx,
            current_y,
            str(val),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            color="black",
        )

ax.tick_params(axis="x", which="both", bottom=False, top=False)

for fmt in ('pdf', 'svg', 'png'):
    plt.savefig(os.path.join(FIG_DIR, f"{FIG_STEM}.{fmt}"),
                format=fmt, transparent=True, dpi=DPI)
plt.close()


# --- REPORT: group sizes, removed outliers, values behind the bars ------------
print(f"[report] Observations: {n_raw} raw -> {n_clean} after per-(Sample, Dox) "
      f"IQR filtering (removed {n_raw - n_clean}).")

for dox_cond, data in (("-", dox_minus_data), ("+", dox_plus_data)):
    print(f"[report] Dox{dox_cond} means +/- SD (ddof=1):")
    for i, sample in enumerate(unique_samples):
        group_vals = df_clean[
            (df_clean["Sample"] == sample) & (df_clean["Dox"] == dox_cond)
        ]["Normalized_Viability"].values
        print(f"[report]   {sample}: {data['Mean_Viability'].iloc[i]:.2f} "
              f"+/- {data['SD'].iloc[i]:.2f} (n={len(group_vals)})")

print(f"[report] Welch t-test, Dox-: S3 vs S7 p={p_s3_s7:.4g} ({stars_s3_s7}); "
      f"S1 vs S7 p={p_s1_s7:.4g} ({stars_s1_s7})")

print(f"[report] Bracket base y={base_y:.3f}, step={step_y}, top ylim="
      f"{base_y + step_y + 19:.3f}.")

for sample, cis_val, trans_val in zip(
        dox_minus_data["Sample"], dox_minus_data["cis-RE"],
        dox_minus_data["trans-RE"]):
    print(f"[report]   {sample}: trans-RE={trans_val}, cis-RE={cis_val}")
