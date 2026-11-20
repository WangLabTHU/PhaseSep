import itertools
import os

import joblib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from scipy import stats

DATA_DIR = './data'
FIG_DIR = './figs'
INPUT_FILES = {
    'argmax': os.path.join(DATA_DIR, 'reporter_argmax.xlsx'),
    'ucb': os.path.join(DATA_DIR, 'reporter_ucb.xlsx'),
}
FIG_STEM = 'F02a_scatters_mean_combined_V8'
TABLE_STEM = 'F02a_sample_statistics_summary_V8'
DPI = 600

# --- NATURE PRESS GLOBAL FONT CONFIGURATION ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
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

selected_samples = ["PSV-4", "PSV-5", "PSV-29", "PSV-53", "PSV-46"]

EXPR_COLS = ["Dox + rep 1", "Dox + rep 2", "Dox + rep 3"]
LEAK_COLS = ["Dox - rep 1", "Dox - rep 2", "Dox - rep 3"]
NORM_COLS = EXPR_COLS + LEAK_COLS
META_COLS = ["Experiment date", "Sample ID", "Sample label", "Trans plasmid ID",
             "group_name", "tetO_num", "Syntax", "Mutant"]

# (table suffix, column suffix, control Trans plasmid ID)
CONTROLS = (("FUS", "", "PSD2-FUS"), ("rtTA", "_rtTA", "PSD0-pwx156"))

palette = {
    "5tetO_21gap": "#32435c",
    "6tetO_20gap": "#7a93ab",
    "6tetO_21gap": "#854c8d",
    "5tetO_17gap": "#bf88b1",
    "6tetO_17gap": "#933d30",
    "7tetO_17gap": "#d37a30",
    "7tetO_16gap": "#e4a57c",
    "8tetO_11gap": "#657d6f",
    "8tetO_10gap": "#3c4e44",
}

shape_mapping = {
    "G10A + G12A": "o",
    "G10A + G12T": "s",
    "C2G + G10A": "^",
    "C2G + G10C": "D",
    "C2G + G10T": "X",
    "C2A + G10A": "*",
    "C8A + G10T": "P",
    "A9T + G10A": "H",
}

plot_config = {
    "expr_cols_norm": [c + "_norm" for c in EXPR_COLS],
    "leak_cols_norm": [c + "_norm" for c in LEAK_COLS],
    "vline": 0.7,
    "xlim": [0.5, 1.05],
    "ylim": [0, 2.05],
}

plot_groups = {
    "argmax_6_7": ("argmax", [5, 6]),
    "argmax_7_8": ("argmax", [7, 8]),
    "ucb_7_8": ("ucb", [7, 8]),
}

title_mapping = {
    "argmax_6_7": "Lib1",
    "argmax_7_8": "Lib2",
    "ucb_7_8": "Lib3",
}

os.makedirs(FIG_DIR, exist_ok=True)

for path in INPUT_FILES.values():
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found; place reporter_argmax.xlsx and reporter_ucb.xlsx in ./data/."
        )

df_dict = {strategy: pd.read_excel(path) for strategy, path in INPUT_FILES.items()}

for strategy, df in df_dict.items():
    missing = [c for c in META_COLS + NORM_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{INPUT_FILES[strategy]} is missing the column(s) {missing}.")
    if df.empty:
        raise ValueError(f"{INPUT_FILES[strategy]} contains no data rows.")

# --- NORMALISATION AGAINST BOTH CONTROLS ---
for strategy, df in df_dict.items():
    for col in ("Syntax", "Mutant"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df[NORM_COLS] = df[NORM_COLS].apply(pd.to_numeric, errors="coerce")

    for exp_date in df["Experiment date"].unique():
        idx = df["Experiment date"] == exp_date
        for _, suffix, trans in CONTROLS:
            control = df[idx & (df["Sample label"] == "clontech")
                         & (df["Trans plasmid ID"] == trans)]
            if control.empty:
                raise ValueError(
                    f"{INPUT_FILES[strategy]}: no {trans} control on {exp_date}."
                )
            expr_mean = np.nanmean(control[EXPR_COLS].values.astype(float))
            leak_mean = np.nanmean(control[LEAK_COLS].values.astype(float))
            if not (np.isfinite(expr_mean) and expr_mean > 0
                    and np.isfinite(leak_mean) and leak_mean > 0):
                raise ValueError(
                    f"{INPUT_FILES[strategy]}: non-positive or missing {trans} "
                    f"control mean on {exp_date}."
                )
            for col in EXPR_COLS:
                df.loc[idx, col + "_norm" + suffix] = df.loc[idx, col] / expr_mean
            for col in LEAK_COLS:
                df.loc[idx, col + "_norm" + suffix] = df.loc[idx, col] / leak_mean


def sem(values):
    values = values[np.isfinite(values)]
    return np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan


def p_value(sample_values, control_values):
    sample_values = sample_values[np.isfinite(sample_values)]
    control_values = control_values[np.isfinite(control_values)]
    return stats.ttest_ind(sample_values, control_values).pvalue


# --- PLOTTING & STATISTICS ---
width_inch = 135 / 25.4
height_inch = 55 / 25.4
fig, axes = plt.subplots(1, 3, figsize=(width_inch, height_inch), sharey=True)

all_stats_records = []

for ax, (fig_name, (strategy, teto_keep)) in zip(axes, plot_groups.items()):
    display_title = title_mapping[fig_name]
    df = df_dict[strategy]

    plot_df = df[df["group_name"].isin([strategy, "control_clontech"])].copy()
    control_df = plot_df[plot_df["group_name"] == "control_clontech"].copy()
    strategy_df = plot_df[
        (plot_df["group_name"] == strategy) & (plot_df["tetO_num"].isin(teto_keep))
    ].copy()
    if strategy_df.empty:
        raise ValueError(f"{strategy}: no rows for tetO_num {teto_keep}.")

    strategy_df["relative_expression"] = strategy_df[plot_config["expr_cols_norm"]].mean(1)
    strategy_df["relative_leakiness"] = strategy_df[plot_config["leak_cols_norm"]].mean(1)
    control_df["relative_expression"] = control_df[plot_config["expr_cols_norm"]].mean(1)
    control_df["relative_leakiness"] = control_df[plot_config["leak_cols_norm"]].mean(1)

    for _, row in strategy_df.iterrows():
        record = {
            "Library": display_title,
            "Sample ID": row["Sample ID"],
            "Experiment Date": row["Experiment date"],
            "Syntax": row.get("Syntax", ""),
            "Mutant": row.get("Mutant", ""),
        }
        for label, suffix, trans in CONTROLS:
            ctrl_match = control_df[
                (control_df["Experiment date"] == row["Experiment date"])
                & (control_df["Trans plasmid ID"] == trans)
            ]
            if ctrl_match.empty:
                raise ValueError(
                    f"{strategy} / {row['Sample ID']}: no {trans} control on "
                    f"{row['Experiment date']}."
                )
            ctrl_row = ctrl_match.iloc[0]
            expr_vals = row[[c + "_norm" + suffix for c in EXPR_COLS]].values.astype(float)
            leak_vals = row[[c + "_norm" + suffix for c in LEAK_COLS]].values.astype(float)
            record[f"Expr_Mean_{label}"] = np.nanmean(expr_vals)
            record[f"Expr_SEM_{label}"] = sem(expr_vals)
            record[f"Leak_Mean_{label}"] = np.nanmean(leak_vals)
            record[f"Leak_SEM_{label}"] = sem(leak_vals)
            record[f"Expr_pval_{label}"] = p_value(
                row[EXPR_COLS].values.astype(float),
                ctrl_row[EXPR_COLS].values.astype(float))
            record[f"Leak_pval_{label}"] = p_value(
                row[LEAK_COLS].values.astype(float),
                ctrl_row[LEAK_COLS].values.astype(float))
        all_stats_records.append(record)

    # Strategy samples
    for (syntax_val, mutant_val, trans_val), tmp in strategy_df.groupby(
            ["Syntax", "Mutant", "Trans plasmid ID"], dropna=False):
        chosen_color = palette.get(syntax_val, "lightgrey")
        chosen_marker = shape_mapping.get(mutant_val, "o")
        ax.scatter(
            tmp["relative_expression"],
            tmp["relative_leakiness"],
            s=12,
            c=chosen_color,
            edgecolors=chosen_color,
            marker=chosen_marker,
            linewidths=0.3,
            alpha=0.85,
            label="_nolegend_",
        )

    # Operator controls, averaged over every experiment date
    for trans_val, color_val in [("PSD2-FUS", "black"), ("PSD0-pwx156", "white")]:
        ctrl_subset = control_df[control_df["Trans plasmid ID"] == trans_val]
        if ctrl_subset.empty:
            continue
        ax.scatter(
            [ctrl_subset["relative_expression"].mean()],
            [ctrl_subset["relative_leakiness"].mean()],
            s=12,
            c=color_val,
            edgecolors="black",
            marker="o",
            linewidths=0.3,
            alpha=0.85,
            label="_nolegend_",
        )

    for _, row in strategy_df.iterrows():
        if str(row["Sample ID"]) in selected_samples:
            ax.text(
                row["relative_expression"] + 0.005,
                row["relative_leakiness"] + 0.005,
                str(row["Sample ID"]),
                fontsize=3.5,
                color="dimgrey",
                va="bottom",
                ha="left",
            )

    if ax is axes[0]:
        ax.set_ylabel("Relative Leakiness (vs. WT with FUS-rtTA)", labelpad=3)
    else:
        ax.set_ylabel("")
    ax.set_title(display_title, fontsize=5, fontweight="bold", pad=5)

    ax.axvline(plot_config["vline"], ls="--", color="grey", linewidth=0.25, alpha=0.6)
    ax.axhline(0.6, ls="--", color="grey", linewidth=0.25, alpha=0.6)
    ax.set_xlim(plot_config["xlim"])
    ax.set_ylim(plot_config["ylim"])
    ax.tick_params(axis="both", which="major", length=1, width=0.25, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.25)

fig.text(0.5, 0.12, "Relative Induced Activity (vs. WT with FUS-rtTA)",
         ha="center", va="center", fontsize=5)


class VerticalHandlerTuple(HandlerTuple):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width,
                       height, fontsize, trans):
        h1, h2 = orig_handle
        a1 = super().create_artists(legend, [h1], xdescent,
                                    ydescent + height * 1.3, width, height,
                                    fontsize, trans)
        a2 = super().create_artists(legend, [h2], xdescent,
                                    ydescent - height * 1.8, width, height,
                                    fontsize, trans)
        return a1 + a2


legend_elements = []
legend_names = []
for m_name, m_shape in shape_mapping.items():
    legend_elements.append(
        mlines.Line2D([], [], color="none", marker=m_shape,
                      markerfacecolor="#4F4F4F", markeredgecolor="#4F4F4F",
                      markersize=4)
    )
    legend_names.append(m_name)

operator_tuple = (
    mlines.Line2D([], [], color="none", marker="o", markerfacecolor="white",
                  markeredgecolor="black", markeredgewidth=0.3, markersize=4),
    mlines.Line2D([], [], color="none", marker="o", markerfacecolor="black",
                  markeredgecolor="black", markeredgewidth=0.3, markersize=4),
)
legend_elements.append(operator_tuple)
legend_names.append("WT operator (with FUS-rtTA)\nWT operator (with rtTA)")

fig.legend(
    handles=legend_elements,
    labels=legend_names,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.99),
    ncol=9,
    handler_map={tuple: VerticalHandlerTuple(ndivide=None)},
    labelspacing=0.3,
    frameon=False,
    fontsize=4.2,
    columnspacing=0.5,
    handletextpad=0.1,
)

fig.subplots_adjust(wspace=0.1, bottom=0.20, top=0.82, left=0.08, right=0.96)

for fmt in ("pdf", "svg", "png"):
    plt.savefig(os.path.join(FIG_DIR, f"{FIG_STEM}.{fmt}"), format=fmt,
                transparent=True, dpi=DPI)
plt.close()

# --- ONE TABLE, BOTH NORMALISATIONS ---
stats_df = pd.DataFrame(all_stats_records)
stats_df.to_csv(os.path.join(FIG_DIR, TABLE_STEM + ".csv"), index=False)
stats_df.to_excel(os.path.join(FIG_DIR, TABLE_STEM + ".xlsx"), index=False)

for label in ("rtTA", "FUS"):
    print(f"\n[{label}] library means")
    print(stats_df.groupby("Library")[
        [f"Expr_Mean_{label}", f"Leak_Mean_{label}"]].mean().round(4).to_string())

    rows = []
    for metric in ("Expr", "Leak"):
        for lib_a, lib_b in itertools.combinations(["Lib1", "Lib2", "Lib3"], 2):
            a = stats_df.loc[stats_df["Library"] == lib_a,
                             f"{metric}_Mean_{label}"].dropna()
            b = stats_df.loc[stats_df["Library"] == lib_b,
                             f"{metric}_Mean_{label}"].dropna()
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            rows.append({
                "Metric": metric,
                "Comparison": f"{lib_a} vs {lib_b}",
                "Mean_1": a.mean(),
                "Mean_2": b.mean(),
                "T": t_stat,
                "p": p_val,
            })
    print(pd.DataFrame(rows).to_string(index=False))
