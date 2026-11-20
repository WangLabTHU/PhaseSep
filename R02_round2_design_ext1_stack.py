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
# deviation 6
from _cgan_arch import (Generator, Discriminator, ResBlock, EncoderLayer,
                        PositionwiseFeedForward)

# The project layout: the round-2 extra arm under ./results, the cGAN's generator under
# ./checks/cgan_hek293t, the predictor under ./checks/round1_design
RESULTS_DIR = './results/round2_design_ext1_stack'

COMBINED_MASKED = os.path.join(RESULTS_DIR, 'condition_masked.txt')

DEEPSEED_CSV = os.path.join(RESULTS_DIR, 'condition_deepseed.csv')
RANDOM_CSV = os.path.join(RESULTS_DIR, 'condition_random.csv')

PREDS_TOTAL_CSV = os.path.join(RESULTS_DIR, 'round2_preds_total.csv')
PREDS_SEL_CSV = os.path.join(RESULTS_DIR, 'round2_preds_sel.csv')
FINAL_CSV = os.path.join(RESULTS_DIR, 'round2_final_ext1.csv')
FINAL_TXT = os.path.join(RESULTS_DIR, 'round2_final_ext1.txt')

# The generator of the round-1 library, and the predictor of the round-1 library
MODEL_NAME = 'deepseed_hek293_debug'
GEN_CKPT = os.path.join(SAVE_ROOT, 'net_G_{}.pth'.format(49999))
CHECKPOINT_DIR = './checks/round1_design/'

# The three branches, and the column each contributes
BRANCHES = ["induced_mpra_means", "uninduced_mpra_means", "lfc"]

# The design: how many times each mask is drawn, and the two thresholds of the selection
N_DRAWS = 100
TOPK = 10

# The draws are seeded per draw from this; see deviation 3
RANDOM_SEED = 42

# The tetO of the round-1 array
WILDTYPE_TETO = "TCCCTATCAGTGATAGAGA"

# The arm's four spacers: the round2_design_p2_extra.py names, in the author's order, and his four
# literals for them. Their names carry their geometry -- 7, 8, 9 or 10 tetOs, a 17, 11, 7 or 5 bp
# gap -- and every one is 252 bp with the wild-type tetO 11 bp from its right end
TEMPLATE_IDS = ["spacer_7tetO_17gap", "spacer_8tetO_11gap", "spacer_9tetO_7gap",
                "spacer_10tetO_5gap"]

TEMPLATES = [
    "MMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMM",
    "MMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMM",
    "MMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMM",
    "MMMMMMTCCCTATCAGTGATAGAGAMMMMMTCCCTATCAGTGATAGAGAMMMMMTCCCTATCAGTGATAGAGAMMMMMTCCCTATCAGTGATAGAGAMMMMMTCCCTATCAGTGATAGAGAMMMMMTCCCTATCAGTGATAGAGAMMMMMTCCCTATCAGTGATAGAGAMMMMMTCCCTATCAGTGATAGAGAMMMMMTCCCTATCAGTGATAGAGAMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMM"
]

# The five tetOs the arm is designed with, the author's five, the same five R02_round2_design.py
# takes from the round-1 screen in descending induced
TETO_IDS = ["mutant_O2_10A_12A", "mutant_O2_10A_12T", "mutant_O2_2G_10A", "mutant_O2_2G_10T",
            "mutant_O2_2G_10C"]
TETO_LIST = ["TCCCTATCAATAATAGAGA", "TCCCTATCAATTATAGAGA", "TGCCTATCAATGATAGAGA",
             "TGCCTATCATTGATAGAGA", "TGCCTATCACTGATAGAGA"]

# The predictor's sequence length, the length of the design, the 19 bp of a tetO, the bases the
# random filler draws from, and where part4 reads the tetO of a design from
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


''' part1: condition_masked -- 四个 spacer x 五个 tetO, 作为掩码 '''

if len(TETO_IDS) != len(TETO_LIST):
    raise ValueError(
        f'{len(TETO_IDS)} tetO ids and {len(TETO_LIST)} tetOs; the arm masks one with the other and '
        f'needs as many of each')

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
            f'part4 reads the tetO of a design from')

with open(COMBINED_MASKED, "w") as f:
    for t_id, template in zip(TEMPLATE_IDS, TEMPLATES):
        for o_id, tetO in zip(TETO_IDS, TETO_LIST):
            new_seq = template.replace(WILDTYPE_TETO, tetO)
            name = f"{t_id}_{o_id}"
            f.write(f">{name}\n{new_seq}\n")

n_masks = len(TEMPLATE_IDS) * len(TETO_IDS)
print(f'[report] {COMBINED_MASKED}: {n_masks} masked templates, the {len(TEMPLATE_IDS)} spacers '
      f'{", ".join(TEMPLATE_IDS)} x the {len(TETO_IDS)} tetOs above')


''' 掩码读回, 掩码名字 -> tetO; 下面两个 draw 段共用 '''

masks, tags = open_fa(COMBINED_MASKED)

# the mask is named spacer_<n>tetO_<gap>gap_<tetO id>, so the last field of the first three
# splits is the tetO id the mask carries
tetO_map = dict(zip(TETO_IDS, TETO_LIST))
mutants = []
for tag in tags:
    tetO_id = tag.split("_", 3)[-1]
    if tetO_id not in tetO_map:
        raise KeyError(
            f'the mask {tag} of {COMBINED_MASKED} carries the tetO {tetO_id}, which is not one of '
            f'the {len(tetO_map)} tetOs of the design; the two come from part1 of this same run')
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


''' part2: step1 -- condition_deepseed, 生成器填掩码, 再把 tetO 放回去 '''

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


''' part3: step2 -- condition_random, 同样的掩码, 空格处随机填充 '''

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


''' part4: 预测模型筛选 -- 三个分支打分, 再按组选 '''

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
print(f'[report] {PREDS_TOTAL_CSV}: {len(df_sorted)} designs, '
      f'{df_sorted["spacer_group"].nunique()} spacer groups')

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
读取round2_final_ext1.csv, 置换为fasta
'''

df = pd.read_csv(FINAL_CSV)
tags_list = df["tags"].tolist()
seqs_list = df["seqs"].tolist()

with open(FINAL_TXT, "w") as f:
    for i in range(len(tags_list)):
        f.write(f">{tags_list[i]}\n{seqs_list[i]}\n")

print(f'[report] {FINAL_TXT}: {len(tags_list)} designs of {DESIGN_LENGTH} bp, the >= 7 tetO arm of '
      f'the round-2 design')
