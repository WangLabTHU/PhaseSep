'''Step 2 of the round-1 library design: build the templates and generate the component libraries.

The second of the three steps that R01_round1_prepare_lib.py used to hold as one file. It writes
the seven component tables that step 3 joins:

  spacer_deepseed    5925  25 draws of the 237 masked spacers, the tetO put back by the mask
  spacer_random      5925  the same spacers with the gaps drawn at random
  tetO_mutant        1596  the wild-type array with one or two bases of the tetO mutated
  flanking_deepseed  1975  the tetO array on a wild-type spacing, once, with the flanks drawn
  flanking_random    1975  the same, drawn at random
  control_positive    454  published inducible designs and the pwx158 array, with repeats
  control_negative    300  100 HEK293T enhancers, 100 without motifs, 100 random sequences

  ./results/round1_library/template/*.txt    the four templates, written first, read by the draws
  ./results/round1_library/*.csv             the seven tables above

Deviations from cgan_debug.py:
  1. paths: the inputs are read from ./data/cgan_template (the four the templates are built from,
     and the published designs) and ./data/cgan_train (the two enhancer files of the negative
     control), the templates and the seven tables are written under ./results/round1_library/, and
     the generator is loaded from ./checks/cgan_hek293t/net_G_49999.pth -- SAVE_ROOT in
     _cgan_arch.py, where the weights are flat: the pretrained file and the one step 1 trains are
     the same path. The original read and wrote ./library/template/, ./library/results/ and
     ./checkpoints/.
  2. the driver here was eight blocks of cgan_debug.py run by hand, seven of them commented out
     once they had been run. All seven are live here, in the author's order, and they run as one
     file: the tables have to be rebuilt from nothing on a new root. The last block of the original
     is step 3, R01_round1_prepare_lib_combine.py.
  3. the templates come from this project's library.py, which is not part of the cgan files and had
     no other home in the three steps. They are built here, first, because every draw below reads
     them. library.py's two fasta reads went through `from gpro.utils.utils_predictor import
     open_fa`; on a root without that package the same records come from open_fa_withidx's sibling
     open_fa in _cgan_arch.py, the author's own reader in cgan_debug.py, and the four files are
     byte for byte the ones the author's run left behind.
  4. tetO_mutant.py is not here. It wrote double_mutant.csv, which was then hand-edited into the
     TetO_mutant.xls that the template builder reads; the xls is an input,
     ./data/cgan_template/TetO_mutant.xls, and nothing upstream of it can be rerun. So is
     positive_published.txt, the 54 published inducible designs that control_positive repeats: the
     author's file, not a table built here.
  5. RANDOM_SEED: the two random libraries, the random half of the two control libraries and the
     sample of the generator were drawn from numpy's global state with no seed, so the run that
     produced the tables on disk cannot be repeated and its random rows cannot be recovered. Every
     draw below is seeded, per draw, from RANDOM_SEED, so this file gives the same library every
     time. The deterministic rows -- the templates, tetO_mutant, the pwx158, clontech and published
     halves of control_positive, and the enhancer and no-motif halves of control_negative -- are
     byte for byte the tables on disk; the rows the randomness touched are a new draw of the same
     kind.
  6. the Levenshtein distance the original computed beside the mask substitution is not computed:
     the substitution never read it. The numbers are unaffected and python-Levenshtein is not a
     dependency.
  7. write_fa does not create the directory it writes into, so the two are made first.
  8. the files this needs are checked before the run rather than failing inside a library, so a
     missing one names itself, with the counts of what was written printed as it goes.
  9. import joblib and the pdf/ps fonttype lines are the standing requirement for delivered files.
 10. the checkpoint is loaded with weights_only=False, the torch 2.6 default refusing whole pickled
     modules; see _cgan_arch.py deviation 6. Because the pretrained net_G_49999.pth is one of those
     modules and was saved by a script that defined the classes itself, it was pickled under the
     name __main__.Generator, and torch.load looks that name up in the running script. The classes
     are therefore imported into this file as well, which is what the last import below is for; a
     checkpoint trained by step 1 of these three files does not need it, having been saved from
     _cgan_arch.

Running this file loads the generator twice -- 25 draws of the 237 masked spacers, then 1975 draws
of the one flanking template -- and spends the rest of its time drawing random sequences, which is
seconds. It prints a [report] line per component table.
'''

import os
import sys
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

import joblib

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# The two cGAN files, which the project copies to the root beside this one
for _library in (os.path.dirname(os.path.abspath(__file__)), './scripts'):
    if _library and os.path.isdir(_library) and _library not in sys.path:
        sys.path.append(_library)

from _cgan_arch import Deepseed, SAVE_ROOT, open_fa, open_fa_withidx, write_fa

# The same classes under this file's name too: the pretrained checkpoint pickled them as
# __main__.Generator, and torch.load looks that name up in the script that is running; see
# deviation 10
from _cgan_arch import (Generator, Discriminator, ResBlock, EncoderLayer,
                        PositionwiseFeedForward)

# The project layout: the cGAN's inputs under ./data, its weights under ./checks/cgan_hek293t, the
# library of the design under ./results
TEMPLATE_DIR = './data/cgan_template'
TRAIN_DIR = './data/cgan_train'
LIB_DIR = './results/round1_library'
TPL_DIR = os.path.join(LIB_DIR, 'template')

# The model of step 1, and the checkpoint that step saved last. The weights are flat in the weights
# directory, so the pretrained file the author's run left behind and the one step 1 trains here are
# the same path: /net_G_49999.pth under SAVE_ROOT
MODEL_NAME = 'cgan_hek293t'
GEN_CKPT = os.path.join(SAVE_ROOT, 'net_G_49999.pth')

# The two random libraries are seeded per draw; see deviation 5
RANDOM_SEED = 42

WILDTYPE_TETO = "TCCCTATCAGTGATAGAGA"
NT_MAP = ["A", "C", "G", "T"]


def require(path, what):
    '''The file has to be where the path says; say which one is missing.'''
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} is missing. {what}')


os.makedirs(TPL_DIR, exist_ok=True)




''' part0: 四个模板 -- 由 library.py 构建 '''

pwx158_seqs = open_fa(os.path.join(TEMPLATE_DIR, "pwx158.fa"))[0]
pwx158_seqs = ("".join(pwx158_seqs)).upper()

wildtype_seqs = open_fa(os.path.join(TEMPLATE_DIR, "7tetO-wild type.txt"))[0]
wildtype_seqs = wildtype_seqs[0].upper()

require(os.path.join(TEMPLATE_DIR, "pwx158.fa"), 'it is one of the template sources of the design')
require(os.path.join(TEMPLATE_DIR, "7tetO-wild type.txt"),
        'it is one of the template sources of the design')
require(os.path.join(TEMPLATE_DIR, "TetO_mutant.xls"),
        'it is the mutation table the tetO template is built from')


def find_first_k_occurrences(seq, pattern, k):

    matches = [m.start() for m in re.finditer(pattern, seq)]
    if len(matches) < k:
        return "Not enough Matches!"
    first_k_index = matches[k-1]
    return first_k_index


## step1: spacer -- positive control

seqs_list = []
tags_list = []


for num_tetO in range(2, 7+1):

    position = find_first_k_occurrences(pwx158_seqs, WILDTYPE_TETO, num_tetO)
    end = position + len(WILDTYPE_TETO) + 11
    start = end - 252
    seqs_list.append(pwx158_seqs[start:end])
    tags_list.append("spacer_{}tetO: positive_control, pwx158".format(num_tetO))

    if (num_tetO == 7):
        continue
    position = find_first_k_occurrences(wildtype_seqs, WILDTYPE_TETO, num_tetO)
    end = position + len(WILDTYPE_TETO) + 11
    rand_seqs = wildtype_seqs[:end]
    rand_seqs = "M" * (252-len(rand_seqs)) + rand_seqs
    seqs_list.append(rand_seqs)
    tags_list.append("spacer_{}tetO: positive_control, random".format(num_tetO))

write_fa(os.path.join(TPL_DIR, "spacer_positive.txt"), seqs_list, tags_list)
print(f'[report] template spacer_positive.txt: {len(seqs_list)} designs')


## step2: spacer -- masked tetO

seqs_list = []
tags_list = []

for num_tetO in range(2, 11+1):
    for spacer in range(3, 50+1):
        if ( len(WILDTYPE_TETO) * num_tetO +  spacer * (num_tetO-1) + 11 > 252):
            continue
        mask_0 = (WILDTYPE_TETO + "M" * spacer) * (num_tetO-1) + WILDTYPE_TETO  + "M" * 11
        mask = "M" * (252-len(mask_0)) + mask_0
        seqs_list.append(mask)
        tags_list.append("spacer_{}tetO: mask_{}gap".format(num_tetO, spacer))

write_fa(os.path.join(TPL_DIR, "spacer_masked.txt"), seqs_list, tags_list)
print(f'[report] template spacer_masked.txt: {len(seqs_list)} designs')


## step3: sequential: wildtype_tetO and random_flanking

tetO_mutants_csv = pd.read_csv(os.path.join(TEMPLATE_DIR, "TetO_mutant.xls"))
mutants = list(tetO_mutants_csv.loc[:,"mutant"])
labels = list(tetO_mutants_csv.loc[:,"label"])

matches = [m.start() for m in re.finditer(WILDTYPE_TETO, wildtype_seqs)]

## tetO_mutant
mutant_list = []
labels_list = []
for i in range(len(mutants)):
    mutant, label = mutants[i], labels[i]

    last_match_end = 0
    tetO_mutant = ""
    for match_start in matches:
        tetO_mutant += wildtype_seqs[last_match_end:match_start]
        tetO_mutant += mutant
        last_match_end = len(tetO_mutant)
    tetO_mutant += wildtype_seqs[last_match_end:]

    mutant_list.append(tetO_mutant)
    labels_list.append("sequential: mutant_tetO, {}".format(label))

write_fa(os.path.join(TPL_DIR, "tetO_mutant.txt"), mutant_list, labels_list)
print(f'[report] template tetO_mutant.txt: {len(mutant_list)} designs')


## flanking_masked
last_match_end = 0
flanking_masked = ""
for match_start in matches:
    flanking_masked += "M" * (match_start - last_match_end)
    flanking_masked += "TCCCTATCAGTGATAGAGA"
    last_match_end = len(flanking_masked)
flanking_masked += "M" * (252 - len(flanking_masked))

write_fa(os.path.join(TPL_DIR, "flanking_masked.txt"), [flanking_masked], ["flanking: wildtype"])
print(f'[report] template flanking_masked.txt: 1 design')


''' part1: spacer_deepseed -- 25次抽样, 每次把tetO用mask覆盖回去 '''

require(GEN_CKPT,
        'it is the generator of step 1, R01_round1_prepare_lib_train.py, or the pretrained one; '
        'the weights sit flat in ' + SAVE_ROOT)
print(f'[report] the generator is {GEN_CKPT}')


masks, tags = open_fa(os.path.join(TPL_DIR, "spacer_masked.txt"))
model = Deepseed(model_name=MODEL_NAME, length=252)

seqs_list = []
tags_list = []


for k in tqdm(range(25)):
    np.random.seed(RANDOM_SEED + k)
    samples = model.generate(input_file=os.path.join(TPL_DIR, "spacer_masked.txt"), sample_model_path=GEN_CKPT, seed=k, sample_output=False)
    tags_k = []

    ## substration
    for i in range(len(samples)):
        matches = [m.start() for m in re.finditer(WILDTYPE_TETO, masks[i])]
        for match in matches:
            samples[i] = samples[i][:match] + masks[i][match:match+19] + samples[i][match+19:]
        prefix = tags[i].split(":")[0]
        suffix = tags[i].split("_")[-1]
        tags_k.append( "{}_{}_deepseed_{}".format(prefix, suffix, k) )

    seqs_list += samples
    tags_list += tags_k

group_list = ["spacer_deepseed"] * len(seqs_list)
df = pd.DataFrame({"seqs": seqs_list, "tags": tags_list, "group": group_list})
df.to_csv(os.path.join(LIB_DIR, "spacer_deepseed.csv"))
print(f'[report] spacer_deepseed: {len(df)} designs, {len(masks)} masked spacers drawn 25 times')


''' part2: spacer_random -- 同样的模板, 空格处随机填充 '''

masks, tags = open_fa(os.path.join(TPL_DIR, "spacer_masked.txt"))
nt_map = NT_MAP

seqs_list = []
tags_list = []

for k in tqdm(range(25)):
    np.random.seed(RANDOM_SEED + k)
    samples = []
    tags_k = []
    for i in range(len(masks)):
        matches = [m.start() for m in re.finditer(WILDTYPE_TETO, masks[i])]
        rand_seq = ""
        last_match_end = 0
        for match in matches:
            nt = np.random.choice(nt_map, size=match-last_match_end, replace=True)
            nt = "".join(nt)
            rand_seq += nt + "TCCCTATCAGTGATAGAGA"
            last_match_end = len(rand_seq)
        nt = np.random.choice(nt_map, size=11, replace=True)
        nt = "".join(nt)
        rand_seq += nt

        samples.append(rand_seq)
        prefix = tags[i].split(":")[0]
        suffix = tags[i].split("_")[-1]
        tags_k.append( "{}_{}_random_{}".format(prefix, suffix, k) )

    seqs_list += samples
    tags_list += tags_k

group_list = ["spacer_random"] * len(seqs_list)
df = pd.DataFrame({"seqs": seqs_list, "tags": tags_list, "group": group_list})
df.to_csv(os.path.join(LIB_DIR, "spacer_random.csv"))
print(f'[report] spacer_random: {len(df)} designs, {len(masks)} masked spacers drawn 25 times')


''' part3: tetO_mutant -- 不做生成, 直接把模板换成表 '''

masks, tags = open_fa(os.path.join(TPL_DIR, "tetO_mutant.txt"))
new_tags = []
for i in range(len(tags)):
    suffix = tags[i].split(",")[1][1:]
    new_tags.append( "mutant_{}".format(suffix) )

group_list = ["tetO_mutant"] * len(masks)
df = pd.DataFrame({"seqs": masks, "tags": new_tags, "group": group_list})
df.to_csv(os.path.join(LIB_DIR, "tetO_mutant.csv"))
print(f'[report] tetO_mutant: {len(df)} designs, the wild-type array with its tetO mutated')


''' part4: flanking_deepseed -- 只有一条模板, 抽样1975次 '''

masks, tags = open_fa(os.path.join(TPL_DIR, "flanking_masked.txt"))
model = Deepseed(model_name=MODEL_NAME, length=252)

seqs_list = []
tags_list = []

masks = masks[0]

for k in tqdm(range(1975)):
    np.random.seed(RANDOM_SEED + k)
    samples = model.generate(input_file=os.path.join(TPL_DIR, "flanking_masked.txt"), sample_model_path=GEN_CKPT, seed=k, sample_output=False)
    samples = samples[0]

    ## substration
    matches = [m.start() for m in re.finditer(WILDTYPE_TETO, masks)]
    for match in matches:
        samples = samples[:match] + masks[match:match+19] + samples[match+19:]

    seqs_list.append(samples)
    tags_list.append("flanking_deepseed_{}".format(k))

group_list = ["flanking_deepseed"] * len(seqs_list)
df = pd.DataFrame({"seqs": seqs_list, "tags": tags_list, "group": group_list})
df.to_csv(os.path.join(LIB_DIR, "flanking_deepseed.csv"))
print(f'[report] flanking_deepseed: {len(df)} designs, one template drawn 1975 times')


''' part5: flanking_random -- 同样的模板, 空格处随机填充 '''

masks, tags = open_fa(os.path.join(TPL_DIR, "flanking_masked.txt"))
nt_map = NT_MAP

seqs_list = []
tags_list = []

masks = masks[0]

for k in tqdm(range(1975)):
    np.random.seed(RANDOM_SEED + k)
    samples = ""
    matches = [m.start() for m in re.finditer(WILDTYPE_TETO, masks)]

    last_match_end = 0
    for match in matches:
        nt = np.random.choice(nt_map, size=match-last_match_end, replace=True)
        nt = "".join(nt)
        samples += nt + "TCCCTATCAGTGATAGAGA"
        last_match_end = len(samples)
    nt = np.random.choice(nt_map, size=11, replace=True)
    nt = "".join(nt)
    samples += nt

    seqs_list.append(samples)
    tags_list.append( "flanking_random_{}".format(k) )

group_list = ["flanking_random"] * len(seqs_list)
df = pd.DataFrame({"seqs": seqs_list, "tags": tags_list, "group": group_list})
df.to_csv(os.path.join(LIB_DIR, "flanking_random.csv"))
print(f'[report] flanking_random: {len(df)} designs, one template drawn 1975 times')


''' part6: control group -- positive '''

require(os.path.join(TEMPLATE_DIR, "positive_published.txt"),
        'it is the published inducible sample that control_positive repeats')

masks, tags = open_fa(os.path.join(TPL_DIR, "spacer_positive.txt"))

seqs_pwx158, seqs_clontech, masks_random = [], [], []
tags_pwx158, tags_clontech, tags_random = [], [], []

for i in range(len(masks)):
    if ("random" in tags[i]):
        masks_random.append(masks[i])
        num_tetO = tags[i].split("_")[1][0]
        tags_random += ["positive_wildtype_random_{}tetO".format(num_tetO)]
    elif ("7tetO" in tags[i]):
        seqs_clontech.append(masks[i])
        tags_clontech += ["positive_wildtype_clontech_7tetO"]
    else:
        seqs_pwx158.append(masks[i])
        num_tetO = tags[i].split("_")[1][0]
        tags_pwx158 += ["positive_wildtype_pwx158_{}tetO".format(num_tetO)]

seqs_pwx158 = [item for item in seqs_pwx158 for _ in range(20)]
tags_pwx158 = ["{}_rep{}".format(item, k) for item in tags_pwx158 for k in range(20)]
seqs_clontech = [item for item in seqs_clontech for _ in range(150)]
tags_clontech = ["{}_rep{}".format(item, k) for item in tags_clontech for k in range(150)]


nt_map = NT_MAP
seqs_list, tags_list = [], []
for k in range(10):
    np.random.seed(RANDOM_SEED + k)
    samples = []
    tags_k = []
    for i in range(len(masks_random)):
        rand_seq = ''.join(np.random.choice(nt_map, replace=True) if base == 'M' else base for base in masks_random[i])
        samples.append(rand_seq)
        tags_k.append( "{}_{}".format(tags_random[i], k) )

    seqs_list += samples
    tags_list += tags_k

seqs_random = [item for item in seqs_list for _ in range(3)]
tags_random = ["{}_rep{}".format(item, k) for item in tags_list for k in range(3)]

seqs_deepseed, tags_deepseed = open_fa(os.path.join(TEMPLATE_DIR, "positive_published.txt"))
seqs_deepseed = [item for item in seqs_deepseed for _ in range(3)]
tags_deepseed = ["{}_rep{}".format(item, k) for item in tags_deepseed for k in range(3)]

seqs_all = seqs_pwx158 + seqs_clontech + seqs_random + seqs_deepseed
tags_all = tags_pwx158 + tags_clontech + tags_random + tags_deepseed
group_all = ["control_positive"] * len(seqs_all)
df = pd.DataFrame({"seqs": seqs_all, "tags": tags_all, "group": group_all})
df.to_csv(os.path.join(LIB_DIR, "control_positive.csv"))
print(f'[report] control_positive: {len(df)} designs, {len(seqs_pwx158)} pwx158, '
      f'{len(seqs_clontech)} clontech, {len(seqs_random)} random, {len(seqs_deepseed)} published')


''' part7: control group -- negative '''

require(os.path.join(TRAIN_DIR, "unmasked_enhancers_HEK293T_240617.txt"),
        'it is where the 100 HEK293T enhancers of the negative control come from')
require(os.path.join(TRAIN_DIR, "no_motif_enhancers_HEK293T_240617.txt"),
        'it is where the 100 no-motif enhancers of the negative control come from')

seqs_hek293_enhancer, _ = open_fa_withidx(os.path.join(TRAIN_DIR, "unmasked_enhancers_HEK293T_240617.txt"))
seqs_hek293_nomotifs, _ = open_fa_withidx(os.path.join(TRAIN_DIR, "no_motif_enhancers_HEK293T_240617.txt"))

seqs_hek293_enhancer = seqs_hek293_enhancer[0:100]
tags_hek293_enhancer = ["negative_hek293_enhancer_{}".format(k) for k in range(100)]
seqs_hek293_nomotifs = seqs_hek293_nomotifs[0:100]
tags_hek293_nomotifs = ["negative_hek293_nomotifs_{}".format(k) for k in range(100)]

nt_map = NT_MAP
seqs_random = []
for k in range(100):
    np.random.seed(RANDOM_SEED + k)
    nt = np.random.choice(nt_map, size=252, replace=True)
    nt = "".join(nt)
    seqs_random.append(nt)
tags_random = ["negative_random_{}".format(k) for k in range(100)]


seqs_all = seqs_hek293_enhancer + seqs_hek293_nomotifs + seqs_random
tags_all = tags_hek293_enhancer + tags_hek293_nomotifs + tags_random
group_all = ["control_negative"] * len(seqs_all)
df = pd.DataFrame({"seqs": seqs_all, "tags": tags_all, "group": group_all})
df.to_csv(os.path.join(LIB_DIR, "control_negative.csv"))
print(f'[report] control_negative: {len(df)} designs, {len(seqs_hek293_enhancer)} HEK293T '
      f'enhancers, {len(seqs_hek293_nomotifs)} without motifs, {len(seqs_random)} random')


print(f'[report] the seven component tables are in {LIB_DIR}; '
      f'R01_round1_prepare_lib_combine.py joins them into the library')
