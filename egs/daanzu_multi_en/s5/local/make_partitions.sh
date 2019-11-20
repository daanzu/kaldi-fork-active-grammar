#!/bin/bash

# Copyright 2016  Allen Guo
#           2017  Xiaohui Zhang
#           2019  David Zurow
# Apache License 2.0

# This script creates the data directories that will be used during training.
# This is discussed fully in README.md, but the gist of it is that the data for each
# stage will be located at data/{MULTI}/{STAGE}, and every training stage is individually
# configurable for maximum flexibility.

# Note: The $stage if-blocks use -eq in this script, so running with --stage 4 will
# run only the stage 4 prep.

multi=multi_a  # This defines the "variant" we're using; see README.md
stage=1

. utils/parse_options.sh

data_dir=data/$multi

mkdir -p $data_dir

# librispeech 100 2k short
if [ $stage -eq 1 ]; then
  utils/subset_data_dir.sh --shortest data/librispeech_100/train 2000 $data_dir/librispeech_100_train_2kshort
  ln -nfs librispeech_100_train_2kshort $data_dir/mono
fi

# librispeech 100 10k
if [ $stage -eq 2 ]; then
  utils/subset_data_dir.sh data/librispeech_100/train 10000 $data_dir/librispeech_100_train_10k
  ln -nfs librispeech_100_train_10k $data_dir/mono_ali
  ln -nfs mono_ali $data_dir/tri1a
  ln -nfs mono_ali $data_dir/tri1a_ali
  ln -nfs mono_ali $data_dir/tri1b
fi

# librispeech 100
if [ $stage -eq 3 ]; then
  ln -nfs ../librispeech_100/train $data_dir/tri1b_ali
  ln -nfs ../librispeech_100/train $data_dir/tri2
fi

# librispeech 100+360
if [ $stage -eq 4 ]; then
  utils/combine_data.sh data/librispeech_combined_460 \
    data/librispeech_{100,360}/train \
    || { echo "Failed to combine data"; exit 1; }
  ln -nfs ../librispeech_combined_460 $data_dir/tri2_ali
  ln -nfs tri2_ali $data_dir/tri3a
fi

if [ $stage -eq 5 ]; then
  utils/combine_data.sh data/librispeech_tedlium \
    data/librispeech_combined_460 data/tedlium/train \
    || { echo "Failed to combine data"; exit 1; }
  ln -nfs ../librispeech_tedlium $data_dir/tri3a_ali
  ln -nfs tri3a_ali $data_dir/tri3b
fi

if [ $stage -eq 6 ]; then
  utils/combine_data.sh data/librispeech_tedlium_cv \
    data/librispeech_tedlium data/cv_{train,dev,validated} \
    || { echo "Failed to combine data"; exit 1; }
  ln -nfs ../librispeech_tedlium_cv $data_dir/tri3b_ali
  ln -nfs tri3b_ali $data_dir/tri4
fi

if [ $stage -eq 7 ]; then
  utils/combine_data.sh data/librispeech_tedlium_cv_speechcommands \
    data/librispeech_tedlium_cv data/speechcommands \
    || { echo "Failed to combine data"; exit 1; }
  ln -nfs ../librispeech_tedlium_cv_speechcommands $data_dir/tri4_ali
  ln -nfs tri4_ali $data_dir/tri5a
fi

if [ $stage -eq 8 ]; then
  utils/combine_data.sh data/librispeech_tedlium_cv_speechcommands_tatoeba \
    data/librispeech_tedlium_cv_speechcommands data/tatoeba \
    || { echo "Failed to combine data"; exit 1; }
  ln -nfs ../librispeech_tedlium_cv_speechcommands_tatoeba $data_dir/tri5a_ali
  ln -nfs tri5a_ali $data_dir/tri5b
fi

if [ $stage -eq 9 ]; then
  utils/combine_data.sh data/librispeech960_tedlium_cv_speechcommands_tatoeba \
    data/librispeech_tedlium_cv_speechcommands_tatoeba data/librispeech_500/train \
    || { echo "Failed to combine data"; exit 1; }
  ln -nfs ../librispeech960_tedlium_cv_speechcommands_tatoeba $data_dir/tri5b_ali
  ln -nfs tri5b_ali $data_dir/tri6a
  ln -nfs tri5b_ali $data_dir/tri6a_ali
fi

# sampled data for ivector extractor training,.etc
if [ $stage -eq 10 ]; then
  ln -nfs tri6a $data_dir/tdnn
  utils/subset_data_dir.sh $data_dir/tdnn \
    100000 $data_dir/tdnn_100k
  utils/subset_data_dir.sh $data_dir/tdnn \
    30000 $data_dir/tdnn_30k
fi

