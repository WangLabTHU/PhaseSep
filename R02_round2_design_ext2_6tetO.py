'''R02_round2_design_6tetO: the 6tetO_17gap control arm of the round-2 design -- one spacer of
six tetOs, the same five tetOs as the round-2 design, and its five designs put under that arm's.

round2_design_p2_6tetO.py in full: the arm's one template masked with five tetOs, the two draws that
fill it, the three branches that score it, and the one design of each spacer group that goes into
the round-2 library beside the >= 7 tetO arm's. The file's own note says what it is for -- "6tetO_
17gap, 做个5条, 直接合并到_p2_extra路径下的表格里, 最后命名为round2_final_ext2" -- a control one tetO
below R02_round2_design_extra.py's four spacers, on the same five tetOs, so that the >= 7 arm can be
read against a spacer of the same library and the same tetOs.

  part0  condition_masked.txt     spacer_6tetO_17gap x the same five tetOs the round-2 design uses
  part1  condition_deepseed.csv   the DeepSEED generator filling the masks
  part2  condition_random.csv     the same masks filled at random
  part3  round2_preds_total.csv   the 1000 designs with the three branches' predictions
         round2_preds_sel.csv     the ten highest induced of each spacer group
  part4  round2_final_ext2.csv       the >= 7 tetO arm's round2_final_ext1.csv with this arm's five under it
         round2_final_ext2.txt       the same rows, as the round-2 library, fasta

Five masked templates -- one spacer, five tetOs -- drawn a hundred times each way, so a spacer group
is one tetO and holds two hundred designs; the ten highest induced of each are kept, and the one of
those ten the model puts lowest on uninduced is what the file's last block adds. The screen is the
round-2 design's own, R02_round2_design.py's part5, as the author's own file has it: this arm is a
control, and its five are put beside the other arm's rows rather than picked among them.

part4 is why the two arms have an order: this file reads the other arm's final table and puts its
own five under it, so R02_round2_design_extra.py has to run first. The two tables are concatenated
as the author concatenated them, so the other arm's rows keep their places at the top; the names and
the sequences are what the table is read for downstream.

Deviations from round2_design_p2_6tetO.py:
  1. paths: the generator is loaded from ./checks/cgan_hek293t/ -- SAVE_ROOT in _cgan_arch.py, the
     weights directory of the cGAN, where the pretrained net_G_49999.pth sits flat -- as
     R02_round2_design.py loads it, and the three branches from ./checks/round1_design/, instead of
     ./checkpoints/check/deepseed_hek293_debug/ and ./checkpoints/round1_design/. Everything written
     goes to ./results/round2_design_ext2_6tetO/, the directory the original wrote to, under the original's own
     file names, round2_final_ext2.csv and _2.txt among them. The arm this file extends keeps its
     ./results/round2_design_ext1_stack/ table, read here as the original reads it.
  2. the two draws are commented out in the original, checked off once they had run -- "after
     cgan_1026" is the note over them -- and what runs here is the round-2 design's own draw code,
     R02_round2_design.py's part3 and part4, with the mask substitution in the generator's output,
     the random filler, the tags and the groups in the order the author had them. The distance the
     restated code computed beside the mask substitution is not computed -- it was never read -- so
     python-Levenshtein is not a dependency of this pipeline.
  3. the draws are seeded. The random library's choice of bases and the one-hot of the masked
     positions inside Deepseed.generate -- backbone_one_hot draws them from numpy's global state,
     which the original left at OS entropy -- are both numpy draws, seeded per draw from
     RANDOM_SEED, and the generator's own draw is seeded by the seed=k that Deepseed.generate is
     handed. N_DRAWS is how many times each mask is drawn: the 100 of R02_round2_design.py, the one
     number of the two loops the original does not record anywhere.
  4. the template is the author's, byte for byte, and is still a literal. It is checked before it is
     used -- 252 bp, six wild-type tetOs as its name says, a 17 bp gap, and the last tetO at
     [-30:-11], which is where part3 reads the tetO of a design from. The five tetOs are the
     author's five, held here as literals as they are in round2_design_p2_6tetO.py and
     round2_design_p2_extra.py; they are the same five R02_round2_design.py takes from the round-1
     screen in descending induced.
  5. the files this needs are checked before the run rather than by read_csv's or torch.load's
     error, so a missing one names itself; so is a mask whose name carries a tetO that is not in
     the design's five, and so is the other arm's final table, which is named as the file to run
     first.
  6. the pretrained generator's checkpoint is a whole pickled module, saved by a script that defined
     the classes itself, so it was pickled under the name __main__.Generator and torch.load looks
     that name up in the script that is running. The classes are imported here as well, which is
     what the second _cgan_arch import below is for; a checkpoint trained by
     R01_round1_prepare_lib_train.py was saved from _cgan_arch and does not need them.
  7. import joblib and the pdf/ps fonttype lines are the standing requirement for delivered files.
     This file draws nothing, so they do nothing here.

Running this file loads the generator N_DRAWS times and runs 2 x N_DRAWS x 5 designs through three
branches; it prints a [report] line per output file.
'''

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

# The project layout: the control arm under ./results/round2_design_ext2_6tetO, the arm it is merged under in
# ./results/round2_design_ext1_stack, the cGAN's generator under ./checks/cgan_hek293t, the predictor under
# ./checks/round1_design
RESULTS_DIR = './results/round2_design_ext2_6tetO'
EXTRA_DIR = './results/round2_design_ext1_stack'

EXTRA_FINAL_CSV = os.path.join(EXTRA_DIR, 'round2_final_ext1.csv')

COMBINED_MASKED = os.path.join(RESULTS_DIR, 'condition_masked.txt')

DEEPSEED_CSV = os.path.join(RESULTS_DIR, 'condition_deepseed.csv')
RANDOM_CSV = os.path.join(RESULTS_DIR, 'condition_random.csv')

PREDS_TOTAL_CSV = os.path.join(RESULTS_DIR, 'round2_preds_total.csv')
PREDS_SEL_CSV = os.path.join(RESULTS_DIR, 'round2_preds_sel.csv')
FINAL2_CSV = os.path.join(RESULTS_DIR, 'round2_final_ext2.csv')
FINAL2_TXT = os.path.join(RESULTS_DIR, 'round2_final_ext2.txt')

# The generator of the round-1 library, and the predictor of the round-1 library
MODEL_NAME = 'deepseed_hek293_debug'
GEN_CKPT = os.path.join(SAVE_ROOT, 'net_G_{}.pth'.format(49999))
CHECKPOINT_DIR = './checks/round1_design/'

# The three branches, and the column each contributes
BRANCHES = ["induced_mpra_means", "uninduced_mpra_means", "lfc"]

# The arm: how many times each mask is drawn, and how many of each group the screen keeps
N_DRAWS = 100
TOPK = 10

# The draws are seeded per draw from this; see deviation 3
RANDOM_SEED = 42

# The tetO of the round-1 array
WILDTYPE_TETO = "TCCCTATCAGTGATAGAGA"

# The arm's one spacer: the name the author picked, one tetO below the >= 7 tetO arm of
# R02_round2_design_extra.py, and his literal for it -- six tetOs, a 17 bp gap, 252 bp with the
# wild-type tetO 11 bp from its right end
TEMPLATE_IDS = ["spacer_6tetO_17gap"]

TEMPLATES = [
    "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMM"
]

# The five tetOs the arm is designed with, the author's five, the same five R02_round2_design.py
# takes from the round-1 screen in descending induced
TETO_IDS = ["mutant_O2_10A_12A", "mutant_O2_10A_12T", "mutant_O2_2G_10A", "mutant_O2_2G_10T",
            "mutant_O2_2G_10C"]
TETO_LIST = ["TCCCTATCAATAATAGAGA", "TCCCTATCAATTATAGAGA", "TGCCTATCAATGATAGAGA",
             "TGCCTATCATTGATAGAGA", "TGCCTATCACTGATAGAGA"]

# The predictor's sequence length, the length of the design, the 19 bp of a tetO, the bases the
# random filler draws from, and where part3 reads the tetO of a design from
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


''' part0: condition_masked -- spacer_6tetO_17gap x 五个 tetO, 作为掩码 '''

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
            f'part3 reads the tetO of a design from')

with open(COMBINED_MASKED, "w") as f:
    for t_id, template in zip(TEMPLATE_IDS, TEMPLATES):
        for o_id, tetO in zip(TETO_IDS, TETO_LIST):
            new_seq = template.replace(WILDTYPE_TETO, tetO)
            name = f"{t_id}_{o_id}"
            f.write(f">{name}\n{new_seq}\n")

n_masks = len(TEMPLATE_IDS) * len(TETO_IDS)
print(f'[report] {COMBINED_MASKED}: {n_masks} masked templates, the {len(TEMPLATE_IDS)} spacer '
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
            f'the {len(tetO_map)} tetOs of the design; the two come from part0 of this same run')
    mutants.append(tetO_map[tetO_id])

require(GEN_CKPT,
        'it is the DeepSEED generator of the round-1 library, R01_round1_prepare_lib_train.py or '
        'the pretrained one; the weights sit flat in ' + SAVE_ROOT)
for _branch in BRANCHES:
    require(os.path.join(CHECKPOINT_DIR, _branch.split('_')[0], 'checkpoint.pth'),
            'it is one of the three branches R01_round1_predictor.py trains; the design is '
            'selected with them')
require(EXTRA_FINAL_CSV,
        'it is the >= 7 tetO arm\'s own final table, and this arm\'s five are added under it; run '
        'R02_round2_design_extra.py first, which writes it to ' + EXTRA_DIR)

print(f'[report] {len(masks)} masks of {len(tetO_map)} tetOs, drawn {N_DRAWS} times each with '
      f'{GEN_CKPT}')


''' part1: step1 -- condition_deepseed, 生成器填掩码, 再把 tetO 放回去 '''

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


''' part2: step2 -- condition_random, 同样的掩码, 空格处随机填充 '''

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
print(f'[report] {RANDOM_CSV}: {len(df)} designs, {len(masks)} drawn {N_DRAWS} times')


''' part3: 预测模型筛选 -- 三个分支打分, 再按组选 '''

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


''' part4: 本轮五条并入 >=7 tetO 臂的 final 表, 再置换为 fasta '''

df_top_best = (
    df_top_all.loc[df_top_all.groupby("spacer_group")["pred_uninduced"].idxmin()]
    .reset_index(drop=True)
)
print(f'[report] the one of each group the model puts lowest on uninduced, {len(df_top_best)} '
      f'designs')

df_top_prior = pd.read_csv(EXTRA_FINAL_CSV)

n_new = len(df_top_best)
df_top_best = pd.concat([df_top_prior, df_top_best])
df_top_best = df_top_best.reset_index(drop=True)
df_top_best.to_csv(FINAL2_CSV, index=False)
print(f'[report] {FINAL2_CSV}: the {len(df_top_prior)} designs of {EXTRA_FINAL_CSV} with the '
      f'{n_new} of this arm under them, {len(df_top_best)} designs')

df = pd.read_csv(FINAL2_CSV)
tags_list = df["tags"].tolist()
seqs_list = df["seqs"].tolist()

with open(FINAL2_TXT, "w") as f:
    for i in range(len(tags_list)):
        f.write(f">{tags_list[i]}\n{seqs_list[i]}\n")

print(f'[report] {FINAL2_TXT}: {len(tags_list)} designs of {DESIGN_LENGTH} bp, the >= 7 tetO arm '
      f'and its 6tetO_17gap control')
