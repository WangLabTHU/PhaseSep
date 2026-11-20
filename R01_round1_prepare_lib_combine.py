import os
import sys

import pandas as pd

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

from _cgan_arch import open_fa, write_fa

# Where step 2 put the seven component tables, and where the library is written
LIB_DIR = './results/round1_library'

# The two flanks of the synthesised oligo: 15 bp on the left, 20 bp on the right
LEFT_FLANK = "ACTGGCCGCTTGACG"
RIGHT_FLANK = "ggtaggcgtgtacggtggg".upper()


def require(path, what):
    '''The file has to be where the path says; say which one is missing.'''
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} is missing. {what}')


''' part1: combination all -- 七张成分表按作者的顺序叠成文库 '''

COMPONENTS = ['spacer_deepseed', 'spacer_random', 'tetO_mutant',
              'flanking_deepseed', 'flanking_random',
              'control_positive', 'control_negative']

for _name in COMPONENTS:
    require(os.path.join(LIB_DIR, _name + '.csv'),
            'R01_round1_prepare_lib_generate.py writes it; run that step first')

spacer_deepseed = pd.read_csv(os.path.join(LIB_DIR, "spacer_deepseed.csv"))
spacer_random = pd.read_csv(os.path.join(LIB_DIR, "spacer_random.csv"))
tetO_mutant = pd.read_csv(os.path.join(LIB_DIR, "tetO_mutant.csv"))
flanking_deepseed = pd.read_csv(os.path.join(LIB_DIR, "flanking_deepseed.csv"))
flanking_random = pd.read_csv(os.path.join(LIB_DIR, "flanking_random.csv"))
control_positive = pd.read_csv(os.path.join(LIB_DIR, "control_positive.csv"))
control_negative = pd.read_csv(os.path.join(LIB_DIR, "control_negative.csv"))

flanking_deepseed = flanking_deepseed.iloc[:1900,:]
flanking_random = flanking_random.iloc[:1900,:]

result = pd.concat([spacer_deepseed, spacer_random, tetO_mutant, flanking_deepseed, flanking_random, control_positive, control_negative])
result = result.loc[:, ["seqs", "tags", "group"] ]
result.to_csv(os.path.join(LIB_DIR, "final_library.csv"))
result_seqs = list(result.loc[:,"seqs"])
result_tags = list(result.loc[:,"tags"])
write_fa(os.path.join(LIB_DIR, "final_library.txt"), result_seqs, result_tags)

for _name, _table in zip(COMPONENTS, [spacer_deepseed, spacer_random, tetO_mutant,
                                      flanking_deepseed, flanking_random,
                                      control_positive, control_negative]):
    print(f'[report] final_library: {len(_table)} designs from {_name}')
print(f'[report] final_library: {len(result)} designs, written to {LIB_DIR}')


''' part2: add_flank -- 两端加回合成序列的两个 flank '''

seqs, tags = open_fa(os.path.join(LIB_DIR, "final_library.txt"))
left_flank = LEFT_FLANK
right_flank = RIGHT_FLANK

if not (len(seqs) == len(tags) == len(result)):
    raise ValueError(
        'final_library.txt came back as {} sequences and {} tags for a table of {} rows; the '
        'flanking library would not line up with it'.format(len(seqs), len(tags), len(result)))

seqs_flank = [ left_flank + seq + right_flank for seq in seqs ]
write_fa(os.path.join(LIB_DIR, "flank_library.txt"), seqs_flank, tags)
print(f'[report] flank_library.txt: {len(seqs_flank)} designs of '
      f'{len(left_flank) + len(result_seqs[0]) + len(right_flank)} bp '
      f'(the designs that came in at 251 bp stay one base shorter)')

df_final = pd.read_csv(os.path.join(LIB_DIR, "final_library.csv"))
group = list(df_final.loc[:,"group"])
df_flank = pd.DataFrame({"seqs": seqs_flank, "tags": tags, "group": group})
df_flank.to_csv(os.path.join(LIB_DIR, "flank_library.csv"))
print(f'[report] flank_library.csv: {len(df_flank)} designs, the designed library; '
      f'the rest of round 1 starts from this table')
