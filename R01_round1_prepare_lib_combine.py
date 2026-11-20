'''Step 3 of the round-1 library design: join the seven component tables into the designed library.

The last of the three steps that R01_round1_prepare_lib.py used to hold as one file. It is the
author's "## step8: combination all" of cgan_debug.py followed by library/add_flank.py: the seven
tables of step 2 are stacked in the order the author stacked them, the first 1900 rows of the two
flanking libraries are kept, and the 15 bp left and 20 bp right flank of the synthesised oligo are
put back on the ends.

  ./results/round1_library/final_library.csv     the designed library, 18000 designs of 252 bp
  ./results/round1_library/final_library.txt     the same, as fasta
  ./results/round1_library/flank_library.csv     the library as it went to the array, 287 bp
  ./results/round1_library/flank_library.txt     the same, as fasta

The 18000 designs are 5925 + 5925 spacers, 1596 mutants, 1900 + 1900 flanking, and the 454 and 300
controls. 54 of the controls are the published sequences as they were published, at 251 bp, so the
flanking library is 285 or 286 bp rather than 287 throughout; the author's final_library.csv has
the same 54.

flank_library.csv is the table the MPRA experiment was run from and the predictor of round 1 was
trained on; the rest of the project starts from it.

Deviations from cgan_debug.py and add_flank.py:
  1. paths: the seven tables are read from ./results/round1_library/, where step 2 wrote them, and
     the four library files are written there too. The original read and wrote ../library/results/
     for the first two and ./results/ next to library/results/ for the flanking pair, the two
     directories being the same one from the different working directories.
  2. pandas 2 removed DataFrame.append, which the original stacked the seven tables with. pd.concat
     of the same seven frames in the same order gives the same rows, and keeps the index each frame
     was read with, so final_library.csv is byte for byte the author's file. `.iloc[:1900, :]`, the
     column order seqs, tags, group and the write_fa are unchanged.
  3. add_flank.py re-read the fasta to get the sequences and the tags, and took the group column
     from the csv again. That is kept, as a check as well: the fasta has to have one record per
     table row, or the flanking library would silently lose or repeat designs.
  4. the seven tables are checked for before the join rather than by read_csv's error, so a missing
     one names itself and says which step writes it.
  5. import joblib and the pdf/ps fonttype lines are the standing requirement for delivered files.

Running this file is seconds; it prints a [report] line per output file.
'''

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
