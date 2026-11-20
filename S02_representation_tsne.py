import os
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'

# Project layout: data in ./data, model predictions in ./preds, figures in ./figs
DATA_DIR = './data'
PREDS_DIR = './preds'
FIG_DIR = './figs'
DPI = 600

FIG_STEM_EVO2 = 'S02_evo2_tsne'
FIG_STEM_SEI = 'S02_sei_tsne'

# The three Sei / Evo2 inference artifacts are looked up in ./preds first (where the
# model outputs live), then ./data, ./figs and the working directory, and the resolved
# path of each one is reported.
ARTIFACT_DIRS = [PREDS_DIR, DATA_DIR, FIG_DIR, '.']
EVO2_EMB_NAME = 'S02_evo2_emb.npy'
SEI_EMB_NAME = 'S02_sei_emb.npy'
SEI_ANNO_NAME = 'S02_anno_sei.csv'

# The merged table is built here and never written to disk: the original wrote it out as
# ./figures/fig2g_evo2_merged.csv and read it back in the next block, which is pointless
# once the two halves run in one process. Same two round-1 mean tables as then -- the
# original read them from ./datasets/round1_origin/7_exp_mean_induced.tsv and
# ..._uninduced.tsv, now the FUS_TO names under ./data.
FILE_INDUCED = '7_exp_mean_FUS_TO_induced.tsv'
FILE_UNINDUCED = '7_exp_mean_FUS_TO_uninduced.tsv'

# Registered deviation from the original: TSNE() was called without a seed, so every
# run of the old script produced a different layout. Set TSNE_SEED = None to get the
# original (unreproducible) behaviour back.
TSNE_SEED = 42

GROUPS_OF_INTEREST = ['spacer', 'mutant', 'flanking']
GROUP_COLORS = {
    'spacer': '#74a892',
    'mutant': '#e5c185',
    'flanking': '#c97b84',
}

# The original drew all analysed sequences in one axes as a 3-colour point cloud (s=30,
# alpha=0.6) and the spacer group, by far the largest, buried the other two. Colouring
# the layout in any other way -- filled density contours for the three groups, or their
# outlines -- mixes the colours wherever the design targets share part of the space,
# which is most of it. So each figure is three panels instead: the same grey cloud of all
# analysed sequences, with one design target on top of it, its own points plus two of its
# own density contours. Nothing is ever drawn on top of a second group's colour.
BACKGROUND_COLOR = '#b0b0b0'
POINT_SIZE = 2.5
POINT_ALPHA = 0.18
GROUP_POINT_ALPHA = 0.5
# seaborn's level for a 2-D KDE is the share of the group's own density mass that falls
# OUTSIDE the contour, so 0.5 draws the contour holding the densest half of the group and
# 0.1 the one holding its densest 90%. Read the other way round -- level = mass enclosed
# -- the outer contour comes out as a small blob inside the core instead. Filled contours
# need the level list to end at 1; these are lines, so each level gets its own call and
# can carry its own style.
KDE_CORE_LEVEL = 0.5
KDE_EXTENT_LEVEL = 0.1
# a dotted line this thin all but disappears once the figure is looked at at screen size,
# so the outer contour is dashed instead. A custom dash pattern -- linestyles=(0, (3.5,
# 1.6)) -- is not an option: seaborn hands it straight to contour, which raises
# "Do not know how to convert [0] to dashes".
KDE_EXTENT_DASHES = 'dashed'
KDE_GRIDSIZE = 200
# The whole drawing layer of a panel -- the grey cloud, the group's own points and both
# of its contours -- is rasterised into a single bitmap, as in S02, so that the vector
# files carry one embedded image per panel and only the text stays an editable object.
# Everything the axes draws itself (patch 1, spines 2.5, ticks 1.5, titles 3) sits above
# RASTER_BELOW_ZORDER and is therefore left as vector. Marking the contour sets with
# set_rasterized() instead does not work: matplotlib ignores it and warns.
RASTER_BELOW_ZORDER = 0
Z_CLOUD, Z_POINTS, Z_EXTENT_CONTOUR, Z_CORE_CONTOUR = -5, -4, -3, -2
# three panels side by side: wider than tall, because the layout is a row of squares
FIGURE_SIZE = (12.8, 4.4)
KNN_K = 10
SILHOUETTE_SAMPLE = 5000


def assign_group(name):
    '''The library group of a sequence name, as in the original script. Only the recipe
    for the merged table uses it; the panels below classify on the syntax prefix.'''
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


def get_syntax_group(name):
    name = str(name).lower()
    if name.startswith("spacer_"):
        return "spacer"
    elif name.startswith("mutant_"):
        return "mutant"
    elif name.startswith("flanking_"):
        return "flanking"
    else:
        return "other"


def find_artifact(name, required=True):
    for directory in ARTIFACT_DIRS:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    if required:
        raise FileNotFoundError(
            f"{name} is missing: looked in {ARTIFACT_DIRS}. The embedding arrays and the Sei "
            f"annotation come from the Evo2 / Sei inference runs and belong in ./preds.")
    return None


def tsne_2d(embeddings):
    tsne = TSNE(n_components=2, random_state=TSNE_SEED)
    return tsne.fit_transform(embeddings)


def plot_embedding(df_plot, title, stem):
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, len(GROUPS_OF_INTEREST), figsize=FIGURE_SIZE,
                             sharex=True, sharey=True)

    # every panel frames the same window on the layout, otherwise the three design
    # targets could not be compared by eye
    x_pad = 0.04 * (df_plot["PC1"].max() - df_plot["PC1"].min())
    y_pad = 0.04 * (df_plot["PC2"].max() - df_plot["PC2"].min())
    axes[0].set_xlim(df_plot["PC1"].min() - x_pad, df_plot["PC1"].max() + x_pad)
    axes[0].set_ylim(df_plot["PC2"].min() - y_pad, df_plot["PC2"].max() + y_pad)

    for ax, group in zip(axes, GROUPS_OF_INTEREST):
        subset = df_plot[df_plot["syntax_group"] == group]

        # the same grey cloud of all analysed sequences in every panel: where a group
        # sits inside it is read off by comparing the panels
        ax.scatter(df_plot["PC1"], df_plot["PC2"], s=POINT_SIZE, c=BACKGROUND_COLOR,
                   alpha=POINT_ALPHA, linewidths=0, zorder=Z_CLOUD)
        ax.scatter(subset["PC1"], subset["PC2"], s=POINT_SIZE, c=GROUP_COLORS[group],
                   alpha=GROUP_POINT_ALPHA, linewidths=0, zorder=Z_POINTS)

        # a group this small, or one collapsed onto a line, has no 2-D density to draw and
        # would take the KDE down with it
        if len(subset) < 3 or subset["PC1"].std() == 0 or subset["PC2"].std() == 0:
            print(f"[report]   [note] {group}: {len(subset)} points, no density contour drawn")
        else:
            sns.kdeplot(data=subset, x="PC1", y="PC2", levels=[KDE_EXTENT_LEVEL], fill=False,
                        color=GROUP_COLORS[group], gridsize=KDE_GRIDSIZE,
                        linestyles=KDE_EXTENT_DASHES, linewidths=1.3, ax=ax,
                        zorder=Z_EXTENT_CONTOUR)
            sns.kdeplot(data=subset, x="PC1", y="PC2", levels=[KDE_CORE_LEVEL], fill=False,
                        color=GROUP_COLORS[group], gridsize=KDE_GRIDSIZE,
                        linestyles='-', linewidths=1.8, ax=ax, zorder=Z_CORE_CONTOUR)

        ax.set_rasterization_zorder(RASTER_BELOW_ZORDER)

        ax.set_title(f'{group} ({len(subset)})', fontsize=12, color=GROUP_COLORS[group])
        # the t-SNE coordinates have no units, so the panels carry ticks but no axis
        # titles, exactly as the original left them
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelsize=7, length=2, width=0.5)
        sns.despine(ax=ax)

    handles = [Line2D([], [], marker='o', markersize=5, linestyle='none',
                      color=BACKGROUND_COLOR, alpha=0.6,
                      label=f'all analysed sequences ({len(df_plot)})'),
               Line2D([], [], color='#4c4c4c', linestyle='-', linewidth=1.8,
                      label=f'densest {int(KDE_CORE_LEVEL * 100)}% of a group'),
               Line2D([], [], color='#4c4c4c', linestyle=KDE_EXTENT_DASHES, linewidth=1.3,
                      label=f'densest {int((1 - KDE_EXTENT_LEVEL) * 100)}% of a group')]

    plt.suptitle(title, fontsize=14)
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    for fmt in ('pdf', 'svg', 'png'):
        plt.savefig(os.path.join(FIG_DIR, f'{stem}.{fmt}'), format=fmt, dpi=DPI, bbox_inches='tight')
    plt.close()


def layout_quality(coordinates, labels, tag):
    '''Descriptive numbers for the layout that is being shown, not for the raw
    representation: how much the design targets separate once they are projected.'''
    silhouette = silhouette_score(coordinates, labels, sample_size=SILHOUETTE_SAMPLE,
                                  random_state=TSNE_SEED)
    neighbours = NearestNeighbors(n_neighbors=KNN_K + 1).fit(coordinates)
    indices = neighbours.kneighbors(coordinates, return_distance=False)[:, 1:]
    purity = (labels[indices] == labels[:, None]).mean()
    print(f"[report]   {tag}: silhouette {silhouette:.3f} "
          f"(sample {SILHOUETTE_SAMPLE}) | {KNN_K}-NN same-group purity {purity:.3f}")


# --- data ---------------------------------------------------------------------

os.makedirs(FIG_DIR, exist_ok=True)

induced_path = os.path.join(DATA_DIR, FILE_INDUCED)
uninduced_path = os.path.join(DATA_DIR, FILE_UNINDUCED)
df_FUS_Trans_induced = pd.read_csv(induced_path, sep="\t", header=None)
df_FUS_Trans_uninduced = pd.read_csv(uninduced_path, sep="\t", header=None)
df_FUS_Trans_induced.columns = ["seqs", "induced_mpra_means", "names"]
df_FUS_Trans_uninduced.columns = ["seqs", "uninduced_mpra_means", "names"]
df_merged = pd.merge(df_FUS_Trans_induced, df_FUS_Trans_uninduced, on=['names', 'seqs'], how='inner')
df_merged = df_merged[["seqs", "names", "induced_mpra_means", "uninduced_mpra_means"]]
df_merged["lfc"] = df_merged["induced_mpra_means"] / df_merged["uninduced_mpra_means"]
df_merged["group"] = df_merged["names"].apply(assign_group)

df_merged["syntax_group"] = df_merged["names"].apply(get_syntax_group)
df_analysis = df_merged[df_merged["syntax_group"].isin(GROUPS_OF_INTEREST)].copy()
# the inner merge keeps the row order of the induced table, so these positions address
# the embedding arrays the way the saved csv did
selected_indices = df_analysis.index.tolist()

counts = df_analysis["syntax_group"].value_counts()
print(f"[report] {FILE_INDUCED} + {FILE_UNINDUCED} merged in memory (not written): "
      f"{induced_path}, {uninduced_path}")
print(f"[report] rows {len(df_merged)}, analysed {len(df_analysis)} "
      f"({', '.join(f'{g} {int(counts.get(g, 0))}' for g in GROUPS_OF_INTEREST)})")


# --- Evo2 representation ------------------------------------------------------

evo2_path = find_artifact(EVO2_EMB_NAME)
evo2_raw = np.load(evo2_path)  # (17417, 1, 512)
print(f"[report] {EVO2_EMB_NAME}: {evo2_path}{evo2_raw.shape}")

# the embedding rows are addressed by the merged table's row numbers, so the two have
# to be the same length; the original slices without checking and would silently draw
# the wrong sequences if they were not
if evo2_raw.ndim != 3 or evo2_raw.shape[1] != 1:
    raise ValueError(f"{EVO2_EMB_NAME} is expected as (rows, 1, features), got {evo2_raw.shape}")
if evo2_raw.shape[0] != len(df_merged):
    raise ValueError(f"{EVO2_EMB_NAME} has {evo2_raw.shape[0]} rows but the merged table has "
                     f"{len(df_merged)}; they have to stay row-aligned")

evo2_emb = evo2_raw.squeeze(axis=1)[selected_indices, :]
evo2_2d = tsne_2d(evo2_emb)
df_analysis["PC1"] = evo2_2d[:, 0]  # holds the t-SNE coordinates, as in the original
df_analysis["PC2"] = evo2_2d[:, 1]
plot_embedding(df_analysis, "TSNE Visualization of Evo2 Embeddings by Syntax Group", FIG_STEM_EVO2)
layout_quality(evo2_2d, df_analysis["syntax_group"].values, 'Evo2')


# --- Sei representation -------------------------------------------------------

sei_path = find_artifact(SEI_EMB_NAME)
sei_anno_path = find_artifact(SEI_ANNO_NAME)
sei_emb = np.load(sei_path)  # (17417, 21907)
df_anno = pd.read_csv(sei_anno_path)
df_anno = df_anno.drop(columns=["index"], errors='ignore').reset_index(drop=True)
print(f"[report] {SEI_EMB_NAME}: {sei_path}{sei_emb.shape}")
print(f"[report] {SEI_ANNO_NAME}: {sei_anno_path} ({len(df_anno)} rows)")

if sei_emb.shape[0] != len(df_merged):
    raise ValueError(f"{SEI_EMB_NAME} has {sei_emb.shape[0]} rows but the merged table has "
                     f"{len(df_merged)}; they have to stay row-aligned")
sei_emb = sei_emb[selected_indices, :]

if len(df_anno) < sei_emb.shape[1]:
    raise ValueError(f"{SEI_ANNO_NAME} has fewer rows than {SEI_EMB_NAME} has channels, so the "
                     f"HEK293 channel selection cannot be aligned: {len(df_anno)} vs {sei_emb.shape[1]}")
if len(df_anno) > sei_emb.shape[1]:
    print(f"[report]   [note] {SEI_ANNO_NAME} is longer than the channel axis, so its last "
          f"{len(df_anno) - sei_emb.shape[1]} rows annotate no channel")

idx = df_anno[df_anno["celltype"].str.contains("HEK293")].index.tolist()
out_of_range = [i for i in idx if i >= sei_emb.shape[1]]
if out_of_range:
    raise ValueError(f"{len(out_of_range)} of the HEK293 rows of {SEI_ANNO_NAME} point past the last "
                     f"of the {sei_emb.shape[1]} channels in {SEI_EMB_NAME}, so the channel selection "
                     f"cannot be aligned (the original would raise a bare IndexError here)")
sei_emb = sei_emb[:, idx]
df_anno = df_anno.loc[idx].reset_index(drop=True)
print(f"[report]   HEK293 channels {len(idx)}")

sei_2d = tsne_2d(sei_emb)
df_analysis["PC1"] = sei_2d[:, 0]
df_analysis["PC2"] = sei_2d[:, 1]
plot_embedding(df_analysis, "TSNE Visualization of Sei Embeddings by Syntax Group", FIG_STEM_SEI)
layout_quality(sei_2d, df_analysis["syntax_group"].values, 'Sei')
