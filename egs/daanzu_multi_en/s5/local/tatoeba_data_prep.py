#!/usr/bin/env python

# Copyright 2019  David Zurow

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import os
import sys
import csv
import subprocess

if len(sys.argv) != 3:
    print ('Usage: python tatoeba_data_prep.py <corpus_dir> <out_dir>')
    sys.exit(1)
corpus_dir = sys.argv[1]
out_dir = sys.argv[2]

subprocess.check_call("mkdir -p " + out_dir, shell=True)

with open(os.path.join(corpus_dir, 'tatoeba_audio_eng', 'sentences_with_audio.csv')) as tsv_file, \
        open(os.path.join(out_dir, 'text'), 'w') as text_file, \
        open(os.path.join(out_dir, 'wav.scp'), 'w') as wav_scp_file, \
        open(os.path.join(out_dir, 'utt2spk'), 'w') as utt2spk_file:

    reader = csv.DictReader(tsv_file, dialect='excel-tab')

    # id username text
    for row in reader:
        sentence_id = row['id']
        text = row['text']
        spk_id = row['username']
        filepath = os.path.join('tatoeba_audio_eng', 'audio', spk_id, sentence_id + '.mp3')
        utt_id = spk_id + '-' + filepath.replace('/', '-')

        if not os.path.isfile(os.path.join(corpus_dir, filepath)):
            continue

        actual_filepath = os.path.join(corpus_dir, filepath)
        if os.path.isfile(actual_filepath) and os.path.getsize(actual_filepath) != 0:
            text_file.write(utt_id + ' ' + text + '\n')
            wav_scp_file.write(utt_id + ' ' + 'sox {}/{} -t wav -r 16k -b 16 -e signed - |'.format(corpus_dir, filepath) + '\n')
            utt2spk_file.write(utt_id + ' ' + spk_id + '\n')

subprocess.check_call("env LC_COLLATE=C utils/fix_data_dir.sh {}".format(out_dir), shell=True)
subprocess.check_call("env LC_COLLATE=C utils/validate_data_dir.sh --no-feats {}".format(out_dir), shell=True)
