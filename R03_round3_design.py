'''R03_round3_design: the round-3 design -- ten predictors instead of one, and the UCB rule that
reads them on the variants of seven and eight tetOs, whose leakiness the round-2 screen's single
predictor is not trusted on.

round3_design_condition_lfc.py, whose body the author had commented out, run live. The two tables
of the screen, the ten bootstrap predictors, the four spacers masked with the five tetOs, the two
draws that fill them and the UCB screen that keeps one design of each spacer group.

  part0  ./results/round3_design/summary_spacer.csv   the 7 and 8 tetO spacers of round 2
         ./results/round3_design/summary_tetO.csv     the round-1 candidate tetOs, by uninduced
  part1  ./data/round3_design/train_seqs_{i}.txt      ten bootstrap draws of the round-1 library
         ./data/round3_design/train_{induced,uninduced,lfc}_{i}.txt
         ./checks/round3_design/ckpt_{i}/             the three branches trained on each
  part2  ./results/round3_design/combined_masked.txt  four spacers x five tetOs
  part3  ./results/round3_design/combined_deepseed.csv  the generator filling the masks
         ./results/round3_design/combined_random.csv    the same masks filled at random
  part4  ./results/round3_design/round3_preds_total.csv  the 4000 designs, ten members a branch
         ./results/round3_design/round3_preds_sel.csv    the ten lowest acq of each spacer group
         ./results/round3_design/round3_final.csv        the highest induced of each group's ten
         ./results/round3_design/round3_final.txt        the same twenty, as the round-3 library

The file's own words for what it is for: "我需要设计一些7~8 tetO (本底表达量上限高), 且lfc也足够高的
样本 / 考虑到这部分数据的'稀缺性', 以及本身高泄露、leakiness预测不准确, 可以使用UCB方法进行采样 /
各做10条, 比如选2x2=4个模板, 每个做5条 (5个tetO, 可以干脆用之前挑好的) / 选出来泄露足够低的模板
(7tetO: 3gap~18gap, 8tetO: 3gap~12gap)". Its own reference for the sampling is round2_design_p1_comb.py
and its data reference is results/round2_design/total_spacer.csv, which is why part0 reads the round-2
spacer table: this is the one design of the library that stands on a previous round's output.

The screen is the author's UCB rule. Every design is read by all ten members of every branch; the
branch's figure is the mean of the ten, and the design's acquisition is the uninduced mean plus two
of its standard deviations -- a design the members disagree on is read as leakier than its mean
says. Each spacer group keeps the ten designs with the lowest acquisition, and of those ten the one
the members put highest on induced, by mean. It is the round-2 screen's shape with acq in place of
induced and the means in place of the single predictor's readings.

Deviations from round3_design_condition_lfc.py:
  1. the body is live. The original held all of it commented out, from the two mean tables down to
     the three round-3 files of the fasta block; what those lines said is what runs, line for line.
     Its helper functions -- open_fa_with_tags, prepare_datasets, train_branch -- are its own and
     run as they were written.
  2. paths: the round-1 mean tables are read from ./data under their renamed project names,
     7_exp_mean_FUS_TO_induced.tsv and 7_exp_mean_FUS_TO_uninduced.tsv. The original read
     ./datasets/round1_origin/7_exp_mean_induced.tsv and _uninduced.tsv, wrote its bootstrap draws
     to ./datasets/round3_design/, and trained into ./checkpoints/round3_design/. Here the draws
     go to ./data/round3_design/, the ten checkpoints to ./checks/round3_design/, and everything
     the design itself writes to ./results/round3_design/, the directory the original wrote to and
     the one fig5a_reporter.py reads round3_final.csv from. The round-2 spacer table is read from
     ./results/round2_design/total_spacer.csv, the path the original read it from, which is where
     R02_round2_design.py writes it -- so that file runs first.
  3. the two mean tables are read once and shared. The original read them twice: step 1's bootstrap
     block reads them for the ten draws, and step 2's screen reads them again for the candidate
     tetOs. Here the merge is made once and both parts take it, so the two cannot come apart.
  4. the two screens the original only printed are written, under the names the round-2 design uses
     for its own -- ./results/round3_design/summary_spacer.csv and summary_tetO.csv. The spacer
     screen is the author's line, the round-2 spacer table filtered to the 7 and 8 tetO spacers:
     those four groups are the design's four templates. The tetO screen is the round-1 candidate
     table, ordered by uninduced ascending, printed down to the candidates below 0.2 uninduced --
     the author's own print, and the list his five tetOs are picked from. That table is built here
     from the round-1 mean tables rather than read from ./results/round1_design/tetO_plots/
     candidate_tetOs_from_MPRA.csv, which this project does not keep: the recipe is
     R02_round2_design.py part1's, no thresholds moved, and only the sort is this file's -- round 3
     reads its candidates by uninduced, round 2 read its own by induced. The two drop('Unnamed: 0')
     lines of the original are not here: the column they dropped was a csv round trip's product,
     and these frames are built in memory.
  5. the two draws are not in the original at all: step 3 reads combined_deepseed.csv and
     combined_random.csv, which its docstring leaves to the generator script, cgan_1026.py, as
     "生成模型". They run here, as R02_round2_design.py's part3 and part4 -- the same code the
     author's own comment in round2_design_p1_comb.py restates for this round, tqdm(range(100)) over
     ./results/round3_design/combined_masked.txt among it -- with the mask substitution in the
     generator's output, the random filler, the tags and the groups as the author had them. The
     distance that restated code computed beside the mask substitution is not computed, it was never
     read, so python-Levenshtein is not a dependency of this pipeline.
  6. the draws are seeded, as the round-2 design's are. The random filler's choice of bases and the
     one-hot of the masked positions inside Deepseed.generate -- backbone_one_hot draws them from
     numpy's global state, which the original left at OS entropy -- are both numpy draws, seeded per
     draw from RANDOM_SEED, and the generator's own draw is seeded by the seed=k handed to
     Deepseed.generate. N_DRAWS is how many times each mask is drawn: the 100 of the author's own
     restated snippet for this round, the one number neither this file nor round 2 records. The
     ensemble's own training is the author's and is not seeded -- the shuffling DataLoader of
     model.py draws from torch's global state, as it did for him -- so the ten members' weights,
     and with them the four files of part4, differ from one run to the next. Everything up to them
     is the seeded part: same masks, same two draws, same ten bootstrap datasets.
  7. the four templates and the five tetOs are the author's, byte for byte, and are checked before
     they are used: 252 bp, as many wild-type tetOs as the name says, the gap the name says, and the
     last tetO at [-30:-11], which is where part4 reads the tetO of a design from. The five tetOs
     are held as literals as the author held them, and are the same five his snippet lists -- rows
     of the candidate table above 1.6 induced and 5.0 lfc under 0.2 uninduced, which part0 prints.
  8. the files this needs are checked before the run rather than by read_csv's or torch.load's
     error, so a missing one names itself; so is a mask whose name carries a tetO that is not in the
     design's five. The round-2 spacer table is named as the file to run first.
  9. the generator's checkpoint is a whole pickled module, saved by a script that defined the
     classes itself, so it was pickled under the name __main__.Generator and torch.load looks that
     name up in the script that is running. The classes are imported here as well, which is what the
     second _cgan_arch import below is for.
 10. the file's own evaluation block is not run: three regplots of the ten members and the NDCG@K
     curve, which the original drew into ./results/round3_design/ from the bootstrap draws it had
     just made. It is a reading of the ensemble, not of the design, and S11_model_evaluation.py
     draws those two kinds of figure for the round-1 branches; the block's own functions are the
     file's only part left out here, and they are the author's.
 11. the re, random, defaultdict and combinations imports are the author's and are kept as they
     were; random, defaultdict and combinations are not used by this file. import joblib and the
     pdf/ps fonttype lines are the standing requirement for delivered files. This file draws
     nothing, so they do nothing here.

Running this file prepares ten bootstrap datasets and trains thirty branches on them, 400 epochs
each, then draws its 4000 designs and screens them; it prints a [report] line per output file.
It reads ./results/round2_design/total_spacer.csv, so R02_round2_design.py runs before it.
'''

from _model import *
import re
import random
from collections import defaultdict
from itertools import combinations

import joblib
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import sys

# The two libraries of the project, copied to the root beside this file: _model.py is the
# AttnBiLSTM predictor, _cgan_arch.py the DeepSEED generator
for _library in (os.path.dirname(os.path.abspath(__file__)), './scripts', './'):
    if _library and os.path.isdir(_library) and _library not in sys.path:
        sys.path.append(_library)

from _cgan_arch import Deepseed, SAVE_ROOT, open_fa

# The same classes under this file's name too: the pretrained checkpoint pickled them as
# __main__.Generator, and torch.load looks that name up in the script that is running; see
# deviation 9
from _cgan_arch import (Generator, Discriminator, ResBlock, EncoderLayer,
                        PositionwiseFeedForward)

# The project layout: the round-1 mean tables and the ten bootstrap draws under ./data, the ten
# checkpoints under ./checks, the design under ./results, the round-2 spacer table it reads under
# ./results/round2_design
DATA_DIR = './data'
DATASET_DIR = './data/round3_design'
CHECKPOINT_DIR = './checks/round3_design/'
RESULTS_DIR = './results/round3_design'

ROUND2_RESULTS_DIR = './results/round2_design'
TOTAL_SPACER_CSV = os.path.join(ROUND2_RESULTS_DIR, 'total_spacer.csv')

SUMMARY_SPACER_CSV = os.path.join(RESULTS_DIR, 'summary_spacer.csv')
SUMMARY_TETO_CSV = os.path.join(RESULTS_DIR, 'summary_tetO.csv')

COMBINED_MASKED = os.path.join(RESULTS_DIR, 'combined_masked.txt')

COMBINED_DEEPSEED_CSV = os.path.join(RESULTS_DIR, 'combined_deepseed.csv')
COMBINED_RANDOM_CSV = os.path.join(RESULTS_DIR, 'combined_random.csv')

PREDS_TOTAL_CSV = os.path.join(RESULTS_DIR, 'round3_preds_total.csv')
PREDS_SEL_CSV = os.path.join(RESULTS_DIR, 'round3_preds_sel.csv')
FINAL_CSV = os.path.join(RESULTS_DIR, 'round3_final.csv')
FINAL_TXT = os.path.join(RESULTS_DIR, 'round3_final.txt')

FILE_FUS_INDUCED = '7_exp_mean_FUS_TO_induced.tsv'
FILE_FUS_UNINDUCED = '7_exp_mean_FUS_TO_uninduced.tsv'

INDUCED_TSV = os.path.join(DATA_DIR, FILE_FUS_INDUCED)
UNINDUCED_TSV = os.path.join(DATA_DIR, FILE_FUS_UNINDUCED)

# The generator of the round-1 library, whose weights sit flat in ./checks/cgan_hek293t
MODEL_NAME = 'cgan_hek293t'
GEN_CKPT = os.path.join(SAVE_ROOT, 'net_G_{}.pth'.format(49999))

# The three branches, and the column each contributes
BRANCHES = ["induced_mpra_means", "uninduced_mpra_means", "lfc"]

# The ensemble: how many members, and the share of the round-1 library each is trained on
BAGGING_NUM = 10
SAMPLE_RATIO = 0.6

# The design: how many times each mask is drawn, and how many of each group the screen keeps
N_DRAWS = 100
TOPK = 10

# The draws are seeded per draw from this; see deviation 6
RANDOM_SEED = 42

# The candidate screen of part0, the same thresholds R02_round2_design.py screens with, and the
# uninduced ceiling this file's own print reads its candidates down to
CAND_MIN_INDUCED = 1.6
CAND_MIN_LFC = 5.0
TETO_MAX_UNINDUCED = 0.2
CLONTECH_NAME = 'positive_wildtype_clontech_7tetO_rep0'

# The tetO of the round-1 array
WILDTYPE_TETO = "TCCCTATCAGTGATAGAGA"

# The design's four spacers: the 7 and 8 tetO spacers of the round-2 spacer table, and the author's
# four literals for them. Their names carry their geometry -- seven or eight tetOs, a 17, 16, 11 or
# 10 bp gap -- and every one is 252 bp with the wild-type tetO 11 bp from its right end
TEMPLATE_IDS = ["spacer_7tetO_17gap", "spacer_7tetO_16gap", "spacer_8tetO_11gap",
                "spacer_8tetO_10gap"]

TEMPLATES = [
    "MMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMM",
    "MMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMM",
    "MMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMM",
    "MMMMMMMMMMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMTCCCTATCAGTGATAGAGAMMMMMMMMMMM"
]

# The five tetOs the design is made with, the author's five -- the low-leak end of the round-1
# candidates, the list his own snippet carries, held here as literals as it holds them
TETO_IDS = ["mutant_O2_2A_10A", "mutant_O2_8A_10T", "mutant_O2_9T_10A", "mutant_O2_10A_12A",
            "mutant_O2_10A_12T"]
TETO_LIST = ["TACCTATCAATGATAGAGA", "TCCCTATAATTGATAGAGA", "TCCCTATCTATGATAGAGA",
             "TCCCTATCAATAATAGAGA", "TCCCTATCAATTATAGAGA"]

# The predictor's sequence length, the 19 bp of a tetO, the bases the random filler draws from, and
# where part4 reads the tetO of a design from
DESIGN_LENGTH = 252
TETO_LENGTH = 19
TRAILING_TETO = slice(-30, -11)
NT_MAP = ["A", "C", "G", "T"]


def require(path, what):
    '''The file has to be where the path says; say which one is missing.'''
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} is missing. {what}')


def open_fa_with_tags(file):
    record, tags = [], []
    f = open(file,'r')
    for item in f:
        if '>' not in item:
            record.append(item[0:-1])
        else:
            tags.append(item[1:-1])
    f.close()
    return record, tags

def prepare_datasets(df_merged):

    return (
        df_merged["seqs"].tolist(),
        df_merged["induced_mpra_means"].tolist(),
        df_merged["uninduced_mpra_means"].tolist(),
        df_merged["lfc"].tolist(),
    )

def train_branch(task_name, seqs_file, labels_file, save_dir, length=252, epoch=400, patience=50):
    os.makedirs(save_dir, exist_ok=True)
    model = AttnBilstm_language(length=length, epoch=epoch, patience=patience, model_name=task_name)
    model.train(dataset=seqs_file, labels=labels_file, savepath=save_dir)


os.makedirs(RESULTS_DIR, exist_ok=True)


''' 第一轮文库的两张 mean 表; part0 与 part1 都读这两张 '''

require(INDUCED_TSV,
        'put the round-1 library mean table there, the induced arm of 7_exp_mean_FUS_TO_*.tsv; it '
        'is what the ensemble trains on and what its candidate tetOs are read from')
require(UNINDUCED_TSV,
        'put the round-1 library mean table there, the uninduced arm of 7_exp_mean_FUS_TO_*.tsv; it '
        'is what the ensemble trains on and what its candidate tetOs are read from')

df_induced = pd.read_csv(INDUCED_TSV, sep="\t", header=None)
df_uninduced = pd.read_csv(UNINDUCED_TSV, sep="\t", header=None)
df_induced.columns = ["seqs", "induced_mpra_means", "names"]
df_uninduced.columns = ["seqs", "uninduced_mpra_means", "names"]

print(f'[report] {len(df_induced)} rows of {INDUCED_TSV}, {len(df_uninduced)} rows of '
      f'{UNINDUCED_TSV}')

df_library = pd.merge(df_induced, df_uninduced, on=['names', 'seqs'], how='inner')
df_library = df_library[["seqs", "names", "induced_mpra_means", "uninduced_mpra_means"]]
df_library["lfc"] = df_library["induced_mpra_means"] / df_library["uninduced_mpra_means"]


''' part0: round3 自己的 summary_spacer 与 summary_tetO '''

# the screen the four round-3 templates were picked from: the round-2 spacer table's 7 and 8 tetO
# spacers, in its own lfc order
require(TOTAL_SPACER_CSV,
        'it is the round-2 spacer table and this arm reads the 7 and 8 tetO spacers out of it; run '
        'R02_round2_design.py first, which writes it to ' + ROUND2_RESULTS_DIR)

df_grouped = pd.read_csv(TOTAL_SPACER_CSV)
df_grouped = df_grouped[df_grouped['label'].str.contains('7tetO|8tetO')]
df_grouped = df_grouped.drop(columns=["Unnamed: 0"]).reset_index(drop=True)
df_grouped.to_csv(SUMMARY_SPACER_CSV)

print(df_grouped.to_csv(), end='')
print(f'[report] {SUMMARY_SPACER_CSV}: the {len(df_grouped)} spacer group(s) of {TOTAL_SPACER_CSV} '
      f'whose name carries 7 or 8 tetOs, the table printed above; they are the design\'s four '
      f'templates')

# the round-1 candidates the five tetOs are picked from, by uninduced ascending
df_merged1 = df_library[df_library["names"].str.startswith("mutant")]
df_merged2 = df_library[df_library["names"] == CLONTECH_NAME]
if len(df_merged2) == 0:
    raise FileNotFoundError(
        f'{CLONTECH_NAME} is not in the round-1 mean tables {INDUCED_TSV} and {UNINDUCED_TSV}, so '
        f'the wild-type control the screen keeps could not be put back and the candidate table '
        f'would carry no reference tetO')
df_merged = pd.concat((df_merged1, df_merged2)).reset_index(drop=True)

# the tetO of a round-1 design is the 19 bp at offset 8 of its 252 bp sequence
df_merged["tetO"] = [seq[8:8+TETO_LENGTH] for seq in df_merged["seqs"].tolist()]

df_tetOs_MPRA = df_merged[(df_merged["induced_mpra_means"] >= CAND_MIN_INDUCED) &
                          (df_merged["lfc"] >= CAND_MIN_LFC)]
df_tetOs_MPRA = pd.concat((df_tetOs_MPRA, df_merged[df_merged["names"] == CLONTECH_NAME]))
df_tetOs_MPRA = df_tetOs_MPRA.sort_values(by="uninduced_mpra_means", ascending=True)
df_tetOs_MPRA = df_tetOs_MPRA.reset_index(drop=True)

df_tetOs_MPRA.to_csv(SUMMARY_TETO_CSV)
print(f'[report] {SUMMARY_TETO_CSV}: {len(df_tetOs_MPRA)} candidate tetOs out of the '
      f'{len(df_merged)} mutant and control rows of the round-1 library, induced >= '
      f'{CAND_MIN_INDUCED} and lfc >= {CAND_MIN_LFC}, ordered by uninduced, clontech kept')

df_tetOs_low = df_tetOs_MPRA[df_tetOs_MPRA["uninduced_mpra_means"] < TETO_MAX_UNINDUCED]

print(df_tetOs_low.to_csv(), end='')
print(f'[report] the {len(df_tetOs_low)} of them under {TETO_MAX_UNINDUCED} uninduced, the table '
      f'printed above; the design takes the author\'s five: '
      + ', '.join(f'{i} {t}' for i, t in zip(TETO_IDS, TETO_LIST)))


''' part1: 预测模型的 bagging 训练 -- 第一轮文库的十次重抽样, 三支各训一个 '''

train_seqs, train_induced, train_uninduced, train_lfc = prepare_datasets(df_library)

bagging_num = BAGGING_NUM
sample_ratio = SAMPLE_RATIO
n_train = len(train_seqs)
indices = np.arange(n_train)

# write_fa and write_exp do not make the directory they are given
os.makedirs(DATASET_DIR, exist_ok=True)

for i in range(bagging_num):
    model_rng = np.random.RandomState(i)
    sample_size = int(sample_ratio * n_train)
    bootstrap_indices = model_rng.choice(
        indices,
        size=sample_size,
        replace=True)

    bag_seqs = [train_seqs[i] for i in bootstrap_indices]
    bag_names = [df_library["names"].tolist()[i] for i in bootstrap_indices]
    bag_induced = [train_induced[i] for i in bootstrap_indices]
    bag_uninduced = [train_uninduced[i] for i in bootstrap_indices]
    bag_lfc = [train_lfc[i] for i in bootstrap_indices]

    write_fa(os.path.join(DATASET_DIR, f"train_seqs_{i}.txt"), bag_seqs, bag_names)
    write_exp(os.path.join(DATASET_DIR, f"train_induced_{i}.txt"), bag_induced)
    write_exp(os.path.join(DATASET_DIR, f"train_uninduced_{i}.txt"), bag_uninduced)
    write_exp(os.path.join(DATASET_DIR, f"train_lfc_{i}.txt"), bag_lfc)

print(f'[report] {DATASET_DIR}: {bagging_num} bootstrap draws of {n_train} designs, {sample_size} '
      f'each with replacement, {sample_ratio} of the round-1 library')

for i in range(bagging_num):
    base_dir = DATASET_DIR
    # train() takes the branch's directory by string concatenation -- checkpoint_root + model_name
    # -- so this one keeps its trailing slash
    checkpoint_dir = f'{CHECKPOINT_DIR}ckpt_{i}/'
    seqs_file = os.path.join(base_dir, f'train_seqs_{i}.txt')

    train_branch(
        task_name="induced",
        seqs_file=seqs_file,
        labels_file=os.path.join(base_dir, f'train_induced_{i}.txt'),
        save_dir=checkpoint_dir
    )

    train_branch(
        task_name="uninduced",
        seqs_file=seqs_file,
        labels_file=os.path.join(base_dir, f'train_uninduced_{i}.txt'),
        save_dir=checkpoint_dir
    )

    train_branch(
        task_name="lfc",
        seqs_file=seqs_file,
        labels_file=os.path.join(base_dir, f'train_lfc_{i}.txt'),
        save_dir=checkpoint_dir
    )

    print(f'[report] ckpt_{i}: the three branches are trained on {seqs_file} and are under '
          f'{checkpoint_dir}')


''' part2: 模板筛选 -- 四个 spacer x 五个 tetO, 作为掩码 '''

if len(TETO_IDS) != len(TETO_LIST):
    raise ValueError(
        f'{len(TETO_IDS)} tetO ids and {len(TETO_LIST)} tetOs; the design masks one with the other '
        f'and needs as many of each')

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
            f'the {len(tetO_map)} tetOs of the design; the two come from part2 of this same run')
    mutants.append(tetO_map[tetO_id])

require(GEN_CKPT,
        'it is the DeepSEED generator of the round-1 library, R01_round1_prepare_lib_train.py or '
        'the pretrained one; the weights sit flat in ' + SAVE_ROOT)
for _i in range(BAGGING_NUM):
    for _branch in BRANCHES:
        require(os.path.join(CHECKPOINT_DIR, f'ckpt_{_i}', _branch.split('_')[0], 'checkpoint.pth'),
                'it is one of the members of the ensemble the design is screened with; part1 of '
                'this same run trains it')

print(f'[report] {len(masks)} masks of {len(tetO_map)} tetOs, drawn {N_DRAWS} times each with '
      f'{GEN_CKPT}, and screened by the {BAGGING_NUM} members under {CHECKPOINT_DIR}')


''' part3: step1/step2 -- combined_deepseed 与 combined_random '''

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
df.to_csv(COMBINED_DEEPSEED_CSV)
print(f'[report] {COMBINED_DEEPSEED_CSV}: {len(df)} designs, {len(masks)} masks drawn {N_DRAWS} '
      f'times')

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
df.to_csv(COMBINED_RANDOM_CSV)
print(f'[report] {COMBINED_RANDOM_CSV}: {len(df)} designs, {len(masks)} masks drawn {N_DRAWS} '
      f'times')


''' part4: step3 -- 条件式生成 + 预测筛选, UCB采样 '''

df_deepseed = pd.read_csv(COMBINED_DEEPSEED_CSV)
df_random = pd.read_csv(COMBINED_RANDOM_CSV)
seqs_deepseed, tags_deepseed = df_deepseed["seqs"].tolist(), df_deepseed["tags"].tolist()
seqs_random, tags_random = df_random["seqs"].tolist(), df_random["tags"].tolist()

seqs = seqs_deepseed + seqs_random
tags = tags_deepseed + tags_random

checkpoint_base = CHECKPOINT_DIR

df = pd.DataFrame({
    "seqs": seqs,
    "tags": tags
})

bagging_num = BAGGING_NUM
for mode in BRANCHES:
    total_pred = []
    for i in range(bagging_num):
        task_name = mode.split('_')[0]  # 'induced' / 'uninduced' / 'lfc'
        model = AttnBilstm_language(model_name=task_name, length=DESIGN_LENGTH)
        model_path = os.path.join(checkpoint_base, f"ckpt_{i}", task_name, "checkpoint.pth")
        pred = model.predict_input(model_path=model_path, inputs=df["seqs"].tolist(), mode="data")
        total_pred.append(pred)
    total_pred = np.array(total_pred).T # (4000, 10)
    total_pred_mean = np.mean(total_pred, axis=-1, keepdims=True)
    total_pred_std = np.std(total_pred, axis=-1, keepdims=True)
    df[f"pred_mean_{task_name}"] = total_pred_mean
    df[f"pred_std_{task_name}"] = total_pred_std
    print(f'[report] pred_mean_{task_name} and pred_std_{task_name}: {len(df)} designs, read by the '
          f'{bagging_num} members under {checkpoint_base}')

df["spacer_group"] = df["tags"].str.replace(r"_(?:deepseed|random)_\d+$", "", regex=True)
cols = ["spacer_group"] + [c for c in df.columns if c != "spacer_group"]
df = df[cols]
df["tetO"] = [item[-30:-11] for item in df["seqs"]]
df["acq"] = df["pred_mean_uninduced"] + 2 * df["pred_std_uninduced"] # UCB acquisition
df_sorted = df.sort_values(by="acq", ascending=True)
df_sorted.to_csv(PREDS_TOTAL_CSV, index=False)
print(f'[report] {PREDS_TOTAL_CSV}: {len(df_sorted)} designs, {df_sorted["spacer_group"].nunique()} '
      f'spacer groups, sorted by acq, lowest first')

topk = TOPK
top_list = []
for group, subdf in df.groupby("spacer_group"):
    subdf_sorted = subdf.sort_values(by="acq", ascending=True).reset_index(drop=True)
    sub_topk = subdf_sorted.head(topk).copy()
    top_list.append(sub_topk)

df_top_all = pd.concat(top_list, ignore_index=True)
df_top_all.to_csv(PREDS_SEL_CSV, index=False)
print(f'[report] {PREDS_SEL_CSV}: the {topk} lowest acq of each spacer group, or all of a group '
      f'when it has fewer than {topk}, {len(df_top_all)} designs')

df_top_all = pd.read_csv(PREDS_SEL_CSV)
df_top_best = (
    df_top_all.loc[df_top_all.groupby("spacer_group")["pred_mean_induced"].idxmax()]
    .reset_index(drop=True)
)
df_top_best.to_csv(FINAL_CSV, index=False)
print(f'[report] {FINAL_CSV}: the one of each group the ensemble puts highest on induced, '
      f'{len(df_top_best)} designs')


'''

读取round3_final.csv, 置换为fasta
'''

df = pd.read_csv(FINAL_CSV)
tags_list = df["tags"].tolist()
seqs_list = df["seqs"].tolist()

with open(FINAL_TXT, "w") as f:
    for i in range(len(tags_list)):
        f.write(f">{tags_list[i]}\n{seqs_list[i]}\n")

print(f'[report] {FINAL_TXT}: {len(tags_list)} designs of {DESIGN_LENGTH} bp, the round-3 library')
