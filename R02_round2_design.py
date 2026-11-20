import os
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import joblib

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# The two libraries of the project, copied to the root beside this file: _model.py is the
# AttnBiLSTM predictor, _cgan_arch.py the DeepSEED generator
for _library in (os.path.dirname(os.path.abspath(__file__)), './scripts', './'):
    if _library and os.path.isdir(_library) and _library not in sys.path:
        sys.path.append(_library)

from _model import AttnBilstm_language
from _cgan_arch import Deepseed, SAVE_ROOT, open_fa

# The same classes under this file's name too: the pretrained checkpoint pickled them as
# __main__.Generator, and torch.load looks that name up in the script that is running; see
# deviation 8
from _cgan_arch import (Generator, Discriminator, ResBlock, EncoderLayer,
                        PositionwiseFeedForward)

# The project layout: the round-1 mean tables under ./data, the round-2 design under ./results,
# the cGAN's generator under ./checks/cgan_hek293t, the predictor under ./checks/round1_design
DATA_DIR = './data'
RESULTS_DIR = './results/round2_design'

FILE_FUS_INDUCED = '7_exp_mean_FUS_TO_induced.tsv'
FILE_FUS_UNINDUCED = '7_exp_mean_FUS_TO_uninduced.tsv'

INDUCED_TSV = os.path.join(DATA_DIR, FILE_FUS_INDUCED)
UNINDUCED_TSV = os.path.join(DATA_DIR, FILE_FUS_UNINDUCED)

TOTAL_SPACER_CSV = os.path.join(RESULTS_DIR, 'total_spacer.csv')
SUMMARY_SPACER_CSV = os.path.join(RESULTS_DIR, 'summary_spacer.csv')
CAND_CSV = os.path.join(RESULTS_DIR, 'summary_tetO.csv')
COMBINED_MASKED = os.path.join(RESULTS_DIR, 'combined_masked.txt')

DEEPSEED_CSV = os.path.join(RESULTS_DIR, 'combined_deepseed.csv')
RANDOM_CSV = os.path.join(RESULTS_DIR, 'combined_random.csv')

PREDS_TOTAL_CSV = os.path.join(RESULTS_DIR, 'round2_preds_total.csv')
PREDS_SEL_CSV = os.path.join(RESULTS_DIR, 'round2_preds_sel.csv')
FINAL_CSV = os.path.join(RESULTS_DIR, 'round2_final.csv')
FINAL_TXT = os.path.join(RESULTS_DIR, 'round2_final.txt')

# The generator of the round-1 library, and the predictor of the round-1 library
MODEL_NAME = 'cgan_hek293t'
GEN_CKPT = os.path.join(SAVE_ROOT, 'net_G_{}.pth'.format(49999))
CHECKPOINT_DIR = './checks/round1_design/'

# The three branches, and the column each contributes
BRANCHES = ["induced_mpra_means", "uninduced_mpra_means", "lfc"]

# The screen the candidate table is: the MPRA calls an induced tetO good above these two
CAND_MIN_INDUCED = 1.6
CAND_MIN_LFC = 5.0

# What the design keeps of it: the candidates above this in induced
TETO_MIN_INDUCED = 2.0

# The design: how many times each mask is drawn, and the two thresholds of the selection
N_DRAWS = 100
TOPK = 10

# The draws are seeded per draw from this; see deviation 3
RANDOM_SEED = 42

# The tetO of the round-1 array, and the control the screen keeps whatever its own numbers are
WILDTYPE_TETO = "TCCCTATCAGTGATAGAGA"
CLONTECH_NAME = 'positive_wildtype_clontech_7tetO_rep0'

# The design's four spacers: the round-1 screen's best, in the author's order. Their names carry
# their geometry -- 5 or 6 tetOs, a 21, 20, 21 or 17 bp gap -- and the templates below are the
# author's literals for exactly those four
TEMPLATE_IDS = ['spacer_5tetO_21gap', 'spacer_6tetO_20gap', 'spacer_6tetO_21gap',
                'spacer_5tetO_17gap']

TEMPLATES = [
    "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMM",
    "MMMMMMMMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMM",
    "MMMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMM",
    "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMM",
]

# The predictor's sequence length, the length of the design, the 19 bp of a tetO, the bases the
# random filler draws from, and where part5 reads the tetO of a design from
DESIGN_LENGTH = 252
TETO_LENGTH = 19
TRAILING_TETO = slice(-30, -11)
NT_MAP = ["A", "C", "G", "T"]


def require(path, what):
    '''The file has to be where the path says; say which one is missing.'''
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} is missing. {what}')


os.makedirs(RESULTS_DIR, exist_ok=True)


''' 第一轮文库的两张 mean 表; part0 与 part1 都读这两张 '''

require(INDUCED_TSV,
        'put the round-1 library mean table there, the induced arm of 7_exp_mean_FUS_TO_*.tsv; it '
        'is what both the screen and the predictor of round 1 read')
require(UNINDUCED_TSV,
        'put the round-1 library mean table there, the uninduced arm of 7_exp_mean_FUS_TO_*.tsv; '
        'it is what both the screen and the predictor of round 1 read')

df_induced = pd.read_csv(INDUCED_TSV, sep="\t", header=None)
df_uninduced = pd.read_csv(UNINDUCED_TSV, sep="\t", header=None)
df_induced.columns = ["seqs", "induced_mpra_means", "names"]
df_uninduced.columns = ["seqs", "uninduced_mpra_means", "names"]

print(f'[report] {len(df_induced)} rows of {INDUCED_TSV}, {len(df_uninduced)} rows of '
      f'{UNINDUCED_TSV}')


''' part0: total_spacer -- 第一轮 spacer 文库按 label 分组, 按 lfc 排序 '''

# the screen the four round-2 templates were picked from: the spacer library, each label's
# induced / uninduced / lfc averaged over its draws, the wild-type control measured off the same
# array put in as the line the spacers are read against
if not (df_induced["names"].str.contains("clontech").any() and
        df_uninduced["names"].str.contains("clontech").any()):
    raise FileNotFoundError(
        f'no row of {INDUCED_TSV} and {UNINDUCED_TSV} has clontech in its name, so the control '
        f'the spacer library is read against is not there; part0 writes the spacers and that '
        f'control and nothing else')
induced_cali = df_induced[df_induced["names"].str.contains("clontech")]["induced_mpra_means"].iloc[0]
uninduced_cali = df_uninduced[df_uninduced["names"].str.contains("clontech")]["uninduced_mpra_means"].iloc[0]
lfc_cali = induced_cali/uninduced_cali

df_spacer = pd.merge(df_induced, df_uninduced, on=['names', 'seqs'], how='inner')
df_spacer = df_spacer[["seqs", "names", "induced_mpra_means", "uninduced_mpra_means"]]
df_spacer["lfc"] = df_spacer["induced_mpra_means"] / df_spacer["uninduced_mpra_means"]

df_spacer = df_spacer[df_spacer["names"].str.startswith("spacer_")]

df_spacer["label"] = df_spacer["names"].apply(lambda x: "_".join(x.split("_")[:3]))
df_grouped = (
    df_spacer.groupby("label")[["induced_mpra_means", "uninduced_mpra_means", "lfc"]]
    .mean()
    .reset_index()
)
df_grouped.loc[len(df_grouped)] = ["clontech", induced_cali, uninduced_cali, lfc_cali]
# the original's line on the groups -- induced above 1.2 and uninduced below 1.2 -- is left where
# it was, off: everything the library holds is printed and written
# 

df_grouped = df_grouped.sort_values("lfc", ascending=False).reset_index(drop=True)
df_grouped.to_csv(TOTAL_SPACER_CSV)
df_grouped = df_grouped[ (df_grouped["induced_mpra_means"] > 1.2) & (df_grouped["uninduced_mpra_means"] < 1.2) ]
df_grouped = df_grouped.reset_index(drop=True)
df_grouped.to_csv(SUMMARY_SPACER_CSV)

print(df_grouped.to_csv(), end='')
print(f'[report] {SUMMARY_SPACER_CSV}: the same table screened to induced above 1.2 and uninduced '
      f'below 1.2, {len(df_grouped)} row(s), the table printed above')

''' part1: candidate_tetOs_from_MPRA -- 第一轮 MPRA 筛出的候选 tetO '''

df_merged = pd.merge(df_induced, df_uninduced, on=['names', 'seqs'], how='inner')
df_merged = df_merged[["seqs", "names", "induced_mpra_means", "uninduced_mpra_means"]]
df_merged["lfc"] = df_merged["induced_mpra_means"] / df_merged["uninduced_mpra_means"]

df_merged1 = df_merged[df_merged["names"].str.startswith("mutant")]
df_merged2 = df_merged[df_merged["names"] == CLONTECH_NAME]
if len(df_merged2) == 0:
    raise FileNotFoundError(
        f'{CLONTECH_NAME} is not in the round-1 mean tables {INDUCED_TSV} and {UNINDUCED_TSV}, so '
        f'the wild-type control the screen keeps could not be put back and the candidate table '
        f'would carry no reference tetO')
df_merged = pd.concat((df_merged1, df_merged2)).reset_index(drop=True)

# the tetO of a round-1 design is the 19 bp at offset 8 of its 252 bp sequence
df_merged["tetO"] = [seq[8:8+TETO_LENGTH] for seq in df_merged["seqs"].tolist()]

df_cand = df_merged[(df_merged["induced_mpra_means"] >= CAND_MIN_INDUCED) &
                    (df_merged["lfc"] >= CAND_MIN_LFC)]
df_cand = pd.concat((df_cand, df_merged[df_merged["names"] == CLONTECH_NAME]))
df_cand = df_cand.sort_values(by="induced_mpra_means", ascending=False)
df_cand = df_cand.reset_index(drop=True)

df_cand.to_csv(CAND_CSV)
print(f'[report] {CAND_CSV}: {len(df_cand)} candidate tetOs out of the {len(df_merged)} mutant '
      f'and control rows of the round-1 library, induced >= {CAND_MIN_INDUCED} and lfc >= '
      f'{CAND_MIN_LFC}, clontech kept')

''' part2: combined_masked -- 四个 spacer x 最强的候选 tetO, 作为掩码 '''

df_tetOs_MPRA = df_cand.sort_values(by="induced_mpra_means", ascending=False).reset_index(drop=True)
df_tetOs_MPRA = df_tetOs_MPRA[df_tetOs_MPRA["induced_mpra_means"] > TETO_MIN_INDUCED]
df_tetOs_MPRA = df_tetOs_MPRA.reset_index(drop=True)

tetO_ids = df_tetOs_MPRA["names"].tolist()
tetOs_list = df_tetOs_MPRA["tetO"].tolist()

# the control row is concat-ed back in whatever its own numbers are, so a control whose own
# numbers also passed the two filters of part1 is in the candidate table twice. The screen the
# design is taken from was the mutants': the four round-2 spacers carry five mutant tetOs, the
# same five round2_design_p2_6tetO.py and p2_extra.py list. So a repeat is not put on the
# templates twice; it is dropped here and said
tetO_map_kept = {}
for o_id, tetO in zip(tetO_ids, tetOs_list):
    tetO_map_kept.setdefault(o_id, tetO)
if len(tetO_map_kept) != len(tetO_ids):
    print(f'[report] the candidate table has {len(tetO_ids)} rows for {len(tetO_map_kept)} tetOs: '
          f'the {CLONTECH_NAME} row passed the two filters of part1 as well as being kept, and '
          f'{len(tetO_ids) - len(tetO_map_kept)} repeat(s) of it are not designed')
tetO_ids = list(tetO_map_kept)
tetOs_list = [tetO_map_kept[o_id] for o_id in tetO_ids]

if len(tetO_ids) == 0:
    raise ValueError(
        f'no candidate tetO of {CAND_CSV} is above {TETO_MIN_INDUCED} in induced, so the design '
        f'would have no tetO to put on its templates; check the two filters of part1 and the '
        f'round-1 mean tables they were applied to')

print(f'[report] the design takes the {len(tetO_ids)} candidates above {TETO_MIN_INDUCED} in '
      f'induced, in descending induced: '
      + ', '.join(f'{i} {t}' for i, t in zip(tetO_ids, tetOs_list)))

for t_id, template in zip(TEMPLATE_IDS, TEMPLATES):
    num_tetO, gap = re.match(r'spacer_(\d+)tetO_(\d+)gap', t_id).groups()
    num_tetO, gap = int(num_tetO), int(gap)
    if len(template) != DESIGN_LENGTH:
        raise ValueError(
            f'the template of {t_id} is {len(template)} bp, not {DESIGN_LENGTH}; the generator and '
            f'the predictor both take {DESIGN_LENGTH} bp sequences')
    if template.count(WILDTYPE_TETO) != num_tetO:
        raise ValueError(
            f'the template of {t_id} carries {template.count(WILDTYPE_TETO)} wild-type tetOs, not '
            f'the {num_tetO} its name says')
    if re.search(WILDTYPE_TETO + "M" * gap + WILDTYPE_TETO, template) is None:
        raise ValueError(
            f'the template of {t_id} does not carry the {gap} bp gap its name says between two '
            f'wild-type tetOs')
    if template[TRAILING_TETO] != WILDTYPE_TETO:
        raise ValueError(
            f'the last wild-type tetO of the template of {t_id} is not at [-30:-11], which is where '
            f'part5 reads the tetO of a design from')

with open(COMBINED_MASKED, "w") as f:
    for t_id, template in zip(TEMPLATE_IDS, TEMPLATES):
        for o_id, tetO in zip(tetO_ids, tetOs_list):
            new_seq = template.replace(WILDTYPE_TETO, tetO)
            name = f"{t_id}_{o_id}"
            f.write(f">{name}\n{new_seq}\n")

n_masks = len(TEMPLATE_IDS) * len(tetO_ids)
print(f'[report] {COMBINED_MASKED}: {n_masks} masked templates, the {len(TEMPLATE_IDS)} spacers '
      f'{", ".join(TEMPLATE_IDS)} x the {len(tetO_ids)} tetOs above')


''' 掩码读回, 掩码名字 -> tetO; 下面两个 draw 段共用 '''

masks, tags = open_fa(COMBINED_MASKED)

# the mask is named spacer_<n>tetO_<gap>gap_<tetO id>, so the last field of the first three
# splits is the tetO id the mask carries
tetO_map = dict(zip(tetO_ids, tetOs_list))
mutants = []
for tag in tags:
    tetO_id = tag.split("_", 3)[-1]
    if tetO_id not in tetO_map:
        raise KeyError(
            f'the mask {tag} of {COMBINED_MASKED} carries the tetO {tetO_id}, which is not one of '
            f'the {len(tetO_map)} candidates above {TETO_MIN_INDUCED} in induced; the two come from '
            f'part1 and part2 of this same run')
    mutants.append(tetO_map[tetO_id])

require(GEN_CKPT,
        'it is the DeepSEED generator of the round-1 library, R01_round1_prepare_lib_train.py or '
        'the pretrained one; the weights sit flat in ' + SAVE_ROOT)
for _branch in BRANCHES:
    require(os.path.join(CHECKPOINT_DIR, _branch.split('_')[0], 'checkpoint.pth'),
            'it is one of the three branches R01_round1_predictor.py trains; the design is '
            'selected with them')

print(f'[report] {len(masks)} masks of {len(tetO_map)} tetOs, drawn {N_DRAWS} times each with '
      f'{GEN_CKPT}')


''' part3: step1 -- spacer_deepseed, 生成器填掩码, 再把 tetO 放回去 '''

model = Deepseed(model_name=MODEL_NAME, length=DESIGN_LENGTH)

seqs_list = []
tags_list = []

for k in tqdm(range(N_DRAWS)):
    np.random.seed(RANDOM_SEED + k)
    samples = model.generate(input_file=COMBINED_MASKED, sample_model_path=GEN_CKPT, seed=k,
                             sample_output=False)
    tags_k = []

    ## substration: the generator wrote over the tetO as well, put the mask back
    for i in range(len(samples)):
        matches = [m.start() for m in re.finditer(mutants[i], masks[i])]
        for match in matches:
            samples[i] = samples[i][:match] + masks[i][match:match+TETO_LENGTH] \
                + samples[i][match+TETO_LENGTH:]
        tags_k.append("{}_deepseed_{}".format(tags[i], k))

    seqs_list += samples
    tags_list += tags_k

group_list = ["spacer_deepseed"] * len(seqs_list)
df = pd.DataFrame({"seqs": seqs_list, "tags": tags_list, "group": group_list})
df.to_csv(DEEPSEED_CSV)
print(f'[report] {DEEPSEED_CSV}: {len(df)} designs, {len(masks)} masks drawn {N_DRAWS} times')


''' part4: step2 -- spacer_random, 同样的掩码, 空格处随机填充 '''

seqs_list = []
tags_list = []

for k in tqdm(range(N_DRAWS)):
    np.random.seed(RANDOM_SEED + k)
    samples = []
    tags_k = []
    for i in range(len(masks)):
        matches = [m.start() for m in re.finditer(mutants[i], masks[i])]
        rand_seq = ""
        last_match_end = 0
        for match in matches:
            nt = np.random.choice(NT_MAP, size=match-last_match_end, replace=True)
            nt = "".join(nt)
            rand_seq += nt + mutants[i]
            last_match_end = len(rand_seq)
        nt = np.random.choice(NT_MAP, size=11, replace=True)
        nt = "".join(nt)
        rand_seq += nt

        samples.append(rand_seq)
        tags_k.append("{}_random_{}".format(tags[i], k))

    seqs_list += samples
    tags_list += tags_k

group_list = ["spacer_random"] * len(seqs_list)
df = pd.DataFrame({"seqs": seqs_list, "tags": tags_list, "group": group_list})
df.to_csv(RANDOM_CSV)
print(f'[report] {RANDOM_CSV}: {len(df)} designs, {len(masks)} masks drawn {N_DRAWS} times')


''' part5: 预测模型筛选 -- 三个分支打分, 再按组选 '''

df_deepseed = pd.read_csv(DEEPSEED_CSV)
df_random = pd.read_csv(RANDOM_CSV)
seqs_deepseed, tags_deepseed = df_deepseed["seqs"].tolist(), df_deepseed["tags"].tolist()
seqs_random, tags_random = df_random["seqs"].tolist(), df_random["tags"].tolist()

seqs = seqs_deepseed + seqs_random
tags = tags_deepseed + tags_random

df = pd.DataFrame({
    "seqs": seqs,
    "tags": tags
})

for mode in BRANCHES:
    subname = mode.split('_')[0]  # 'induced' / 'uninduced' / 'lfc'
    model = AttnBilstm_language(model_name=subname, length=DESIGN_LENGTH)
    model_path = os.path.join(CHECKPOINT_DIR, subname, "checkpoint.pth")
    preds = model.predict_input(model_path=model_path, inputs=df["seqs"].tolist(), mode="data")
    df[f"pred_{subname}"] = preds
    print(f'[report] pred_{subname}: {len(preds)} designs scored with {model_path}')

df["spacer_group"] = df["tags"].str.replace(r"_(?:deepseed|random)_\d+$", "", regex=True)
cols = ["spacer_group"] + [c for c in df.columns if c != "spacer_group"]
df = df[cols]
df["tetO"] = [item[-30:-11] for item in df["seqs"]]
df_sorted = df.sort_values(by="pred_induced", ascending=False)
df_sorted.to_csv(PREDS_TOTAL_CSV, index=False)
print(f'[report] {PREDS_TOTAL_CSV}: {len(df_sorted)} designs, {df_sorted["spacer_group"].nunique()} '
      f'spacer groups')

topk = TOPK
top_list = []
for group, subdf in df.groupby("spacer_group"):
    subdf_sorted = subdf.sort_values(by="pred_induced", ascending=False).reset_index(drop=True)
    sub_topk = subdf_sorted.head(topk).copy()
    top_list.append(sub_topk)

df_top_all = pd.concat(top_list, ignore_index=True)
df_top_all.to_csv(PREDS_SEL_CSV, index=False)
print(f'[report] {PREDS_SEL_CSV}: the {topk} highest induced of each spacer group, or all of a '
      f'group when it has fewer than {topk}, {len(df_top_all)} designs')

df_top_best = (
    df_top_all.loc[df_top_all.groupby("spacer_group")["pred_uninduced"].idxmin()]
    .reset_index(drop=True)
)
df_top_best.to_csv(FINAL_CSV, index=False)
print(f'[report] {FINAL_CSV}: the one of each group the model puts lowest on uninduced, '
      f'{len(df_top_best)} designs')


'''
读取round2_final.csv, 置换为fasta
'''

df = pd.read_csv(FINAL_CSV)
tags_list = df["tags"].tolist()
seqs_list = df["seqs"].tolist()

with open(FINAL_TXT, "w") as f:
    for i in range(len(tags_list)):
        f.write(f">{tags_list[i]}\n{seqs_list[i]}\n")

print(f'[report] {FINAL_TXT}: {len(tags_list)} designs of {DESIGN_LENGTH} bp, the round-2 design')
