import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'

# The library of the model: _model beside this file (refs_dqx/model.py is byte for byte
# refs_dqx/scripts/lib_attnbilstm.py, and both take their helpers -- open_fa, open_exp, seq2onehot,
# EarlyStopping -- from utils.py). ./scripts and the project root are looked in too, so the file
# also runs from inside the reference tree.
for _library in (os.path.dirname(os.path.abspath(__file__)), './scripts', './'):
    if _library and os.path.isdir(_library) and _library not in sys.path:
        sys.path.append(_library)

try:
    from _model import AttnBilstm_language, seq2onehot
except ImportError:
    from model import AttnBilstm_language, seq2onehot

# The project layout: the tables in ./data, the weights in ./checks, the figures in ./figs
DATA_DIR = './data'
FIG_DIR = './figs'
FIG_STEM = 'S10_tetO_contributions'
DPI = 300

# The round-1 predictor of the FUS arm, trained by R01_round1_predictor.py
CHECKPOINT_DIR = './checks/round1_design/'

FILE_FUS_INDUCED = '7_exp_mean_FUS_TO_induced.tsv'
FILE_FUS_UNINDUCED = '7_exp_mean_FUS_TO_uninduced.tsv'
FILE_tetR_INDUCED = '7_exp_mean_TO_induced.tsv'
FILE_tetR_UNINDUCED = '7_exp_mean_TO_uninduced.tsv'

# What the model is asked about, and how. All six panels score the induced branch, as the original
# did, and SmoothGrad is seeded -- see deviation 7.
MODEL_NAME = 'induced'
N_SAMPLES = 12
NOISE_LEVEL = 0.08
SMOOTHGRAD_SEED = 42

# The scatter the stripplot puts over each box is jittered, and the jitter comes from numpy's
# global stream, which seaborn does not seed and which starts from the operating system. Without
# this the six panels are drawn at six new sets of positions on every run and no two renders of
# the figure are the same. The attribution is reproducible without it -- checked, bit for bit,
# between two processes -- so this is what the figure needs to be.
JITTER_SEED = 42

LENGTH = 252
BATCH_SIZE = 64

# The two wild-type tetO the original works with: the one it searches for, and the reference the
# mutant library is read against
tetO_seq = "TCCCTATCAGTGATAGAGA"
CLONTECH_TAG = "positive_wildtype_clontech_7tetO"

# The colours of the originals: #e5c185 was the high, or best, panel of every pair, #74a892 the low
COLOR_HIGH = "#e5c185"
COLOR_LOW = "#74a892"


def require(path, what):
    '''The file has to be where the path says; say which one is missing.'''
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} is missing. {what}')


def get_syntax_group(name):
    name = str(name).lower()
    if name.startswith("spacer_"):
        return "spacer"
    elif name.startswith("mutant_"):
        return "mutant"
    elif name.startswith("flanking_"):
        return "flanking"
    elif name.startswith("positive_"):
        return "positive"
    elif name.startswith("negative_"):
        return "negative"
    else:
        return "other"


# The four measured tables, the two arms each merged and calibrated by its own negative control,
# and the two arms joined on seqs and names. Only the syntax group and the two induced and
# uninduced columns of the FUS arm are read afterwards.

for _file in (FILE_FUS_INDUCED, FILE_FUS_UNINDUCED, FILE_tetR_INDUCED, FILE_tetR_UNINDUCED):
    require(os.path.join(DATA_DIR, _file), 'put the round-1 library mean tables there')

df_FUS_Trans_induced = pd.read_csv(os.path.join(DATA_DIR, FILE_FUS_INDUCED), sep="\t", header=None)
df_FUS_Trans_uninduced = pd.read_csv(os.path.join(DATA_DIR, FILE_FUS_UNINDUCED), sep="\t", header=None)
df_FUS_Trans_induced.columns = ["seqs", "induced_mpra_means", "names"]
df_FUS_Trans_uninduced.columns = ["seqs", "uninduced_mpra_means", "names"]

df_tetR_Trans_induced = pd.read_csv(os.path.join(DATA_DIR, FILE_tetR_INDUCED), sep="\t", header=None)
df_tetR_Trans_uninduced = pd.read_csv(os.path.join(DATA_DIR, FILE_tetR_UNINDUCED), sep="\t", header=None)
df_tetR_Trans_induced.columns = ["seqs", "induced_mpra_means", "names"]
df_tetR_Trans_uninduced.columns = ["seqs", "uninduced_mpra_means", "names"]

df_FUS_Trans_merged = pd.merge(df_FUS_Trans_induced, df_FUS_Trans_uninduced, on=['names', 'seqs'], how='inner')
df_FUS_Trans_merged = df_FUS_Trans_merged[["seqs", "names", "induced_mpra_means", "uninduced_mpra_means"]]
df_tetR_Trans_merged = pd.merge(df_tetR_Trans_induced, df_tetR_Trans_uninduced, on=['names', 'seqs'], how='inner')
df_tetR_Trans_merged = df_tetR_Trans_merged[["seqs", "names", "induced_mpra_means", "uninduced_mpra_means"]]

df_FUS_Trans_merged["lfc"] = df_FUS_Trans_merged["induced_mpra_means"] / df_FUS_Trans_merged["uninduced_mpra_means"]
df_tetR_Trans_merged["lfc"] = df_tetR_Trans_merged["induced_mpra_means"] / df_tetR_Trans_merged["uninduced_mpra_means"]

row_FUS_Trans_cali = df_FUS_Trans_merged[df_FUS_Trans_merged["names"].str.contains("negative_random", na=False)].mean(numeric_only=True)
row_tetR_Trans_cali = df_tetR_Trans_merged[df_tetR_Trans_merged["names"].str.contains("negative_random", na=False)].mean(numeric_only=True)
cali_induced_FUS_Trans = row_FUS_Trans_cali["induced_mpra_means"]
cali_uninduced_FUS_Trans = row_FUS_Trans_cali["uninduced_mpra_means"]
cali_lfc_FUS_Trans = row_FUS_Trans_cali["lfc"]
df_FUS_Trans_merged["induced_mpra_means"] = df_FUS_Trans_merged["induced_mpra_means"] / cali_induced_FUS_Trans
df_FUS_Trans_merged["uninduced_mpra_means"] = df_FUS_Trans_merged["uninduced_mpra_means"] / cali_uninduced_FUS_Trans
df_FUS_Trans_merged["lfc"] = df_FUS_Trans_merged["lfc"] / cali_lfc_FUS_Trans

cali_induced_tetR_Trans = row_tetR_Trans_cali["induced_mpra_means"]
cali_uninduced_tetR_Trans = row_tetR_Trans_cali["uninduced_mpra_means"]
cali_lfc_tetR_Trans = row_tetR_Trans_cali["lfc"]
df_tetR_Trans_merged["induced_mpra_means"] = df_tetR_Trans_merged["induced_mpra_means"] / cali_induced_tetR_Trans
df_tetR_Trans_merged["uninduced_mpra_means"] = df_tetR_Trans_merged["uninduced_mpra_means"] / cali_uninduced_tetR_Trans
df_tetR_Trans_merged["lfc"] = df_tetR_Trans_merged["lfc"] / cali_lfc_tetR_Trans

df_merged = pd.merge(
    df_FUS_Trans_merged,
    df_tetR_Trans_merged,
    on=["seqs", "names"],
    suffixes=("_FUS", "_tetR")
)
df_merged["syntax_group"] = df_merged["names"].apply(get_syntax_group)
df_mutant = df_merged[df_merged["syntax_group"].isin(["mutant"])].reset_index(drop=True)
df_spacer = df_merged[df_merged["syntax_group"].isin(["spacer"])].reset_index(drop=True)
df_flanking = df_merged[df_merged["syntax_group"].isin(["flanking"])].reset_index(drop=True)

# The two panels of the mutant library are read at the seven positions the clontech control has,
# because a mutant has no wild-type tetO left to find. The control is therefore needed before the
# attribution runs, not when the boxes are drawn.
_clontech = df_merged[df_merged["names"].str.contains(CLONTECH_TAG, na=False)]
if len(_clontech) == 0:
    raise FileNotFoundError(
        f'no row of the round-1 tables is tagged {CLONTECH_TAG}, the control the mutant panels '
        'are read against; it is one of the controls of the round-1 library')
CLONTECH_SEQ = _clontech.iloc[0]["seqs"]
CLONTECH_TETO_STARTS = []
_start_search = 0
for _ in range(7):
    _pos = CLONTECH_SEQ.find(tetO_seq, _start_search)
    if _pos == -1:
        raise ValueError(
            f'the control {CLONTECH_TAG} does not carry seven copies of {tetO_seq}; found '
            f'{len(CLONTECH_TETO_STARTS)}')
    CLONTECH_TETO_STARTS.append(_pos)
    _start_search = _pos + 1
print(f"[report] the {CLONTECH_TAG} control carries its seven tetO at {CLONTECH_TETO_STARTS}")


# The attribution: twelve noisy copies of the input, the mean gradient of the induced model at each
# of them, times the input

def smooth_grad_new(x_batch, model, n_samples=12, noise_level=0.08):
    """适配 Conv1d 输入要求的 SmoothGrad"""
    accumulated_grad = torch.zeros_like(x_batch)
    model.bilstm.rnn.train()

    for _ in range(n_samples):
        noise = torch.randn_like(x_batch) * noise_level
        x_noisy = (x_batch + noise).detach().requires_grad_(True)
        x_input = x_noisy.permute(0, 2, 1)        # (B, L, 4) -> (B, 4, L)
        output = model(x_input)
        if output.dim() > 1:
            output = output.squeeze(dim=-1)

        grads = torch.autograd.grad(
            outputs=output,
            inputs=x_noisy,
            grad_outputs=torch.ones_like(output),
            retain_graph=False,
            create_graph=False
        )[0]
        accumulated_grad += grads
    attr = (accumulated_grad / n_samples) * x_batch
    return attr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_position_attr(X_np, model_name=MODEL_NAME, batch_size=BATCH_SIZE):
    model = AttnBilstm_language(model_name=model_name, length=LENGTH).model
    model_path = os.path.join(CHECKPOINT_DIR, model_name, "checkpoint.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    position_attrs = []
    for start in range(0, len(X_np), batch_size):
        end = min(start + batch_size, len(X_np))
        x_batch = torch.tensor(X_np[start:end], dtype=torch.float32, device=device)
        attr = smooth_grad_new(x_batch, model, n_samples=N_SAMPLES, noise_level=NOISE_LEVEL)
        pos_attr = attr.sum(dim=-1).detach().cpu().numpy()   # (B, L)
        position_attrs.append(pos_attr)
        if start % 200 == 0:
            print(f"[report] processed {start}/{len(X_np)}")
    return np.concatenate(position_attrs, axis=0)


# The two ways the seven boxes are filled. The spacer and the flanking libraries carry the
# wild-type tetO, so their copies are found in the sequence itself; the mutant library does not,
# so its panels are read at the seven positions the clontech reference has.

def draw_tetO_boxplot(ax, df_plot, title, color, jitter):
    # hue = the same seven ids, only so that the seven boxes can take the colour without the
    # palette-without-hue warning; legend off, so the picture is the one the author drew.
    # showfliers off as asked: the open circles the box draws for the points beyond the whiskers
    # repeat the scatter the stripplot already puts over the same box, and every one of those
    # points is still there in the strip. The whiskers are untouched -- they are the 1.5 IQR rule
    # either way.
    sns.boxplot(ax=ax, data=df_plot, x="tetO_id", y="contribution",
                hue="tetO_id", palette=[color] * 7, legend=False, linewidth=1.2,
                showfliers=False)
    sns.stripplot(ax=ax, data=df_plot, x="tetO_id", y="contribution",
                  color="black", alpha=0.5, size=3, jitter=jitter)
    # seaborn writes the two column names, tetO_id and contribution, as the axis labels; the
    # figure writes its own, once, on the edge panels, so they are cleared here
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=14, pad=8)
    ax.tick_params(axis='both', labelsize=12)
    return df_plot


def plot_spacer_tetO_boxplot(ax, pos_attr, df, title, color=COLOR_HIGH, jitter=0.2):
    contrib_data = []

    for i, row in df.iterrows():
        seq = row['seqs']
        seq_name = row['names']
        tetO_positions = []
        start_search = 0
        while True:
            pos = seq.find(tetO_seq, start_search)
            if pos == -1:
                break
            tetO_positions.append(pos)
            start_search = pos + 1

        for tetO_id in range(7):
            if tetO_id >= len(tetO_positions):
                print(f"[report] warning: {seq_name} only found {len(tetO_positions)} tetO")
                break
            start_pos = tetO_positions[tetO_id]
            end_pos = start_pos + len(tetO_seq)
            if end_pos > pos_attr.shape[1]:
                continue
            contrib = pos_attr[i, start_pos:end_pos].sum()

            contrib_data.append({
                "tetO_id": tetO_id + 1,
                "contribution": float(contrib),
                "seq_name": seq_name
            })

    return draw_tetO_boxplot(ax, pd.DataFrame(contrib_data), title, color, jitter)


def plot_mutant_tetO_boxplot(ax, pos_attr, df, title, color=COLOR_HIGH, jitter=0.25):
    contrib_data = []
    tetO_len = len(tetO_seq)

    for i, row in df.iterrows():
        seq_name = row['names']
        for tetO_id, start_pos in enumerate(CLONTECH_TETO_STARTS):
            end_pos = start_pos + tetO_len
            if end_pos > pos_attr.shape[1]:
                continue
            contrib = pos_attr[i, start_pos:end_pos].sum()
            contrib_data.append({
                "tetO_id": tetO_id + 1,
                "contribution": float(contrib),
                "seq_name": seq_name
            })

    if not contrib_data:
        raise ValueError('no contribution data collected for the mutant panels')

    return draw_tetO_boxplot(ax, pd.DataFrame(contrib_data), title, color, jitter)


# The six panels: the spacer library needs no selection, the other two are the 200 highest or
# lowest induced of their library and, of those, the 50 lowest or highest uninduced

pattern = r'^(spacer)_(\d+)tetO_(\d+)gap_(.+)$'
df_spacer[['prefix', 'spacer_ntetO', 'spacer_gap', 'spacer_tag']] = df_spacer['names'].str.extract(pattern)
df_spacer['spacer_ntetO'] = pd.to_numeric(df_spacer['spacer_ntetO'])
df_spacer['spacer_gap'] = pd.to_numeric(df_spacer['spacer_gap'])
df_spacer = df_spacer.drop(columns=['prefix'])

df_spacer_7 = df_spacer[df_spacer['spacer_ntetO'] == 7].copy()
df_high = df_spacer_7[df_spacer_7['spacer_gap'] == 17].reset_index(drop=True)   # 高 induced
df_low  = df_spacer_7[df_spacer_7['spacer_gap'] == 14].reset_index(drop=True)   # 低 induced

df_mutant = df_mutant.copy()
df_good = df_mutant.nlargest(200, 'induced_mpra_means_FUS') \
                  .nsmallest(50, 'uninduced_mpra_means_FUS') \
                  .reset_index(drop=True)
df_bad = df_mutant.nsmallest(200, 'induced_mpra_means_FUS') \
                 .nlargest(50, 'uninduced_mpra_means_FUS') \
                 .reset_index(drop=True)

df_flanking = df_flanking.copy()
df_good_flank = df_flanking.nlargest(200, 'induced_mpra_means_FUS') \
                          .nsmallest(50, 'uninduced_mpra_means_FUS') \
                          .reset_index(drop=True)
df_bad_flank = df_flanking.nsmallest(200, 'induced_mpra_means_FUS') \
                         .nlargest(50, 'uninduced_mpra_means_FUS') \
                         .reset_index(drop=True)

require(os.path.join(CHECKPOINT_DIR, MODEL_NAME, "checkpoint.pth"),
        'put the induced branch of the round-1 predictor there, from R01_round1_predictor.py')

print(f"[report] {len(df_high)} designs at gap 17 and {len(df_low)} at gap 14, "
      f"{len(df_good)} best and {len(df_bad)} worst mutants, "
      f"{len(df_good_flank)} best and {len(df_bad_flank)} worst flankings")

# an empty panel would fail much later, inside the attribution, with nothing to concatenate
for _panel, _frame in (('(a) spacer gap 17', df_high), ('(d) spacer gap 14', df_low),
                       ('(b) mutant best', df_good), ('(e) mutant worst', df_bad),
                       ('(c) flanking best', df_good_flank), ('(f) flanking worst', df_bad_flank)):
    if len(_frame) == 0:
        raise ValueError(
            f'panel {_panel} selected no design out of the round-1 tables; check that both gap 17 '
            'and gap 14 spacers, mutants and flankings are in ./data and that the names follow '
            'spacer_7tetO_17gap_*, mutant_* and flanking_*')

torch.manual_seed(SMOOTHGRAD_SEED)

pos_attr_high = compute_position_attr(seq2onehot(df_high["seqs"].astype(str).tolist(), LENGTH), MODEL_NAME)
pos_attr_low = compute_position_attr(seq2onehot(df_low["seqs"].astype(str).tolist(), LENGTH), MODEL_NAME)
pos_attr_good = compute_position_attr(seq2onehot(df_good["seqs"].astype(str).tolist(), LENGTH), MODEL_NAME)
pos_attr_bad = compute_position_attr(seq2onehot(df_bad["seqs"].astype(str).tolist(), LENGTH), MODEL_NAME)
pos_attr_good_flank = compute_position_attr(seq2onehot(df_good_flank["seqs"].astype(str).tolist(), LENGTH), MODEL_NAME)
pos_attr_bad_flank = compute_position_attr(seq2onehot(df_bad_flank["seqs"].astype(str).tolist(), LENGTH), MODEL_NAME)


# Two rows of three: row 0 the high, or best, member of a pair, row 1 the low, or worst; the
# columns the three libraries. The y axis is shared within a column, so a pair is read on one scale.

np.random.seed(JITTER_SEED)   # see JITTER_SEED: the jitter of the stripplot is what it fixes
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey='col')

records = [
    (axes[0][0], "(a) spacer\ngap=17, high induced",
     plot_spacer_tetO_boxplot(axes[0][0], pos_attr_high, df_high, "(a) spacer\ngap=17, high induced", COLOR_HIGH, 0.2)),
    (axes[0][1], "(b) mutant\nbest (high induced)",
     plot_mutant_tetO_boxplot(axes[0][1], pos_attr_good, df_good, "(b) mutant\nbest (high induced)", COLOR_HIGH, 0.25)),
    (axes[0][2], "(c) flanking\nbest (high induced)",
     plot_spacer_tetO_boxplot(axes[0][2], pos_attr_good_flank, df_good_flank, "(c) flanking\nbest (high induced)", COLOR_HIGH, 0.25)),
    (axes[1][0], "(d) spacer\ngap=14, low induced",
     plot_spacer_tetO_boxplot(axes[1][0], pos_attr_low, df_low, "(d) spacer\ngap=14, low induced", COLOR_LOW, 0.2)),
    (axes[1][1], "(e) mutant\nworst (low induced)",
     plot_mutant_tetO_boxplot(axes[1][1], pos_attr_bad, df_bad, "(e) mutant\nworst (low induced)", COLOR_LOW, 0.25)),
    (axes[1][2], "(f) flanking\nworst (low induced)",
     plot_spacer_tetO_boxplot(axes[1][2], pos_attr_bad_flank, df_bad_flank, "(f) flanking\nworst (low induced)", COLOR_LOW, 0.25)),
]

# the axis labels once, on the edge panels, and no x tick labels under the top row
for _ax, _title, _df_plot in records[:3]:
    _ax.tick_params(labelbottom=False)
for _col in range(3):
    axes[1][_col].set_xlabel("tetO ID (1 to 7)", fontsize=13)
for _row in range(2):
    axes[_row][0].set_ylabel("Sum of Attribution Scores", fontsize=13)

sns.despine()
plt.tight_layout()

os.makedirs(FIG_DIR, exist_ok=True)
for fmt in ('png', 'svg', 'pdf'):
    plt.savefig(os.path.join(FIG_DIR, f'{FIG_STEM}.{fmt}'), format=fmt, dpi=DPI, bbox_inches="tight")
plt.close()


# --- REPORT: the size of every panel, the scale of every pair, the seven medians of every panel ---

print()
print("[report] the model is the round-1 induced predictor, "
      f"{os.path.join(CHECKPOINT_DIR, MODEL_NAME, 'checkpoint.pth')}")
for _row, _row_name in ((0, 'high / best'), (1, 'low / worst')):
    print(f"[report] row {_row}, {_row_name}")
    for _col, (_ax, _title, _df_plot) in enumerate(records[_row * 3:(_row + 1) * 3]):
        medians = _df_plot.groupby("tetO_id")["contribution"].median()
        n_seq = _df_plot["seq_name"].nunique()
        print(f"[report]   {_title.splitlines()[0]:<12s} {n_seq:>4d} designs, "
              f"{len(_df_plot):>5d} boxes   y {_ax.get_ylim()[0]:>9.3f} .. {_ax.get_ylim()[1]:>9.3f}")
        print("[report]     median per tetO 1..7: "
              + "  ".join(f"{medians.get(k, float('nan')):>9.3f}" for k in range(1, 8)))
for _col, _name in enumerate(('spacer', 'mutant', 'flanking')):
    _hi = [m for m in records[0 * 3 + _col][2].groupby("tetO_id")["contribution"].median()]
    _lo = [m for m in records[1 * 3 + _col][2].groupby("tetO_id")["contribution"].median()]
    print(f"[report] {_name:<9s} best-median-minus-worst-median per tetO: "
          + "  ".join(f"{a - b:>9.3f}" for a, b in zip(_hi, _lo)))
