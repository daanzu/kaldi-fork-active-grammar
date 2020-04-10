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
import glob
import re
import subprocess

if len(sys.argv) != 3:
    print ('Usage: python speechcommands_data_prep.py <corpus_dir> <out_dir>')
    sys.exit(1)
corpus_dir = sys.argv[1]
out_dir = sys.argv[2]

subprocess.check_call("mkdir -p " + out_dir, shell=True)

words = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
    'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off',
    'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two',
    'up', 'visual', 'wow', 'yes', 'zero',
]

with open(os.path.join(out_dir, 'text'), 'w') as text_file, \
        open(os.path.join(out_dir, 'wav.scp'), 'w') as wav_scp_file, \
        open(os.path.join(out_dir, 'utt2spk'), 'w') as utt2spk_file:

    for word in words:
        for filepath in glob.glob(os.path.join(corpus_dir, word, '*.wav')):
            text = word
            match = re.match(r'(\w+)_nohash_(\d+).wav', os.path.basename(filepath))
            spk_id = match.group(1)
            utt_id = spk_id + '-' + word + '-' + os.path.basename(filepath)
            
            text_file.write(utt_id + ' ' + text + '\n')
            wav_scp_file.write(utt_id + ' ' + 'sox {} -t wav -r 16k -b 16 -e signed - |'.format(filepath) + '\n')
            utt2spk_file.write(utt_id + ' ' + spk_id + '\n')

subprocess.check_call("env LC_COLLATE=C utils/fix_data_dir.sh {}".format(out_dir), shell=True)
subprocess.check_call("env LC_COLLATE=C utils/validate_data_dir.sh --no-feats {}".format(out_dir), shell=True)
