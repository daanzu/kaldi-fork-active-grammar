#!/usr/bin/env bash

# Implementation of CAPIO Dense CNN-bLSTM (https://arxiv.org/pdf/1801.00059.pdf)
# Based off of mini_librispeech/run_tdnn_1j
# Copyright 2020 David Zurow

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
decode_nj=10
train_set=train_clean_5
test_sets=dev_clean_2
gmm=tri3b
nnet3_affix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=   # affix for the TDNN directory name
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# training options
# training chunk-options
chunk_width=140,100,160
common_egs_dir=
xent_regularize=0.1

# training options
srand=0
remove_egs=true
reporting_email=

#decode options
test_online_decoding=true  # if true, it will run the last decoding stage.


# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 11" if you have already
# run those things.
local/nnet3/run_ivector_common.sh --stage $stage \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --nnet3-affix "$nnet3_affix" || exit 1;

# Problem: We have removed the "train_" prefix of our training set in
# the alignment directory names! Bad!
gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/dense_cnn_blstm${affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 10 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 11 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 75 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 12 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from
  # the GMM baseline.
   if [ -f $tree_dir/final.mdl ]; then
     echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
     exit 1;
  fi
  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 3500 ${lores_train_data_dir} \
    $lang $ali_dir $tree_dir
fi


if [ $stage -le 13 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

  ivector_affine_opts="l2-regularize=0.01"
  cnn_opts="l2-regularize=0.01"
  # lstm_opts l2-regularize=0.0005?
  lstm_opts="decay-time=20 dropout-proportion=0.0"
  lstm_dim1=256
  lstm_dim2=64
  output_opts="l2-regularize=0.015"
  label_delay=5

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # # this takes the MFCCs and generates filterbank coefficients.  The MFCCs
  # # are more compressible so we prefer to dump the MFCCs to disk rather
  # # than filterbanks.
  # idct-layer name=idct input=input dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat
  # batchnorm-component name=batchnorm0 input=idct
  # spec-augment-layer name=spec-augment freq-max-proportion=0.5 time-zeroed-proportion=0.2 time-mask-max-frames=20
  
  # delta-layer name=delta input=spec-augment
  # no-op-component name=input2 input=Append(delta, Scale(0.4, ReplaceIndex(ivector, t, 0)))

  # this takes the MFCCs and generates filterbank coefficients.  The MFCCs
  # are more compressible so we prefer to dump the MFCCs to disk rather
  # than filterbanks.
  idct-layer name=idct input=input dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat

  linear-component name=ivector-linear $ivector_affine_opts dim=200 input=ReplaceIndex(ivector, t, 0)
  batchnorm-component name=ivector-batchnorm target-rms=0.025

  batchnorm-component name=idct-batchnorm input=idct
  combine-feature-maps-layer name=combine_inputs input=Append(idct-batchnorm, ivector-batchnorm) num-filters1=1 num-filters2=5 height=40

  conv-relu-batchnorm-layer name=cnn1 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64 
  conv-relu-batchnorm-layer name=cnn2 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=cnn3 $cnn_opts height-in=40 height-out=20 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=cnn4 $cnn_opts height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=cnn5 $cnn_opts height-in=20 height-out=10 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  conv-relu-batchnorm-layer name=cnn6 $cnn_opts height-in=10 height-out=10  time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults

  fast-lstmp-layer name=blstm-n1m1-forward input=cnn6 cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm-n1m1-backward input=cnn6 cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=3 $lstm_opts
  fast-lstmp-layer name=blstm-n1m2-forward input=Append(blstm-n1m1-forward, blstm-n1m1-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm-n1m2-backward input=Append(blstm-n1m1-forward, blstm-n1m1-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=3 $lstm_opts
  fast-lstmp-layer name=blstm-n1m3-forward input=Append(blstm-n1m1-forward, blstm-n1m1-backward, blstm-n1m2-forward, blstm-n1m2-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm-n1m3-backward input=Append(blstm-n1m1-forward, blstm-n1m1-backward, blstm-n1m2-forward, blstm-n1m2-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=3 $lstm_opts
  fast-lstmp-layer name=blstm-n1m4-forward input=Append(blstm-n1m1-forward, blstm-n1m1-backward, blstm-n1m2-forward, blstm-n1m2-backward, blstm-n1m3-forward, blstm-n1m3-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm-n1m4-backward input=Append(blstm-n1m1-forward, blstm-n1m1-backward, blstm-n1m2-forward, blstm-n1m2-backward, blstm-n1m3-forward, blstm-n1m3-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=3 $lstm_opts
  fast-lstmp-layer name=blstm-n1m5-forward input=Append(blstm-n1m1-forward, blstm-n1m1-backward, blstm-n1m2-forward, blstm-n1m2-backward, blstm-n1m3-forward, blstm-n1m3-backward, blstm-n1m4-forward, blstm-n1m4-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm-n1m5-backward input=Append(blstm-n1m1-forward, blstm-n1m1-backward, blstm-n1m2-forward, blstm-n1m2-backward, blstm-n1m3-forward, blstm-n1m3-backward, blstm-n1m4-forward, blstm-n1m4-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=3 $lstm_opts
  fast-lstmp-layer name=blstm-n1m6-forward input=Append(blstm-n1m1-forward, blstm-n1m1-backward, blstm-n1m2-forward, blstm-n1m2-backward, blstm-n1m3-forward, blstm-n1m3-backward, blstm-n1m4-forward, blstm-n1m4-backward, blstm-n1m5-forward, blstm-n1m5-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm-n1m6-backward input=Append(blstm-n1m1-forward, blstm-n1m1-backward, blstm-n1m2-forward, blstm-n1m2-backward, blstm-n1m3-forward, blstm-n1m3-backward, blstm-n1m4-forward, blstm-n1m4-backward, blstm-n1m5-forward, blstm-n1m5-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=3 $lstm_opts
  fast-lstmp-layer name=blstm-n1m7-forward input=Append(blstm-n1m1-forward, blstm-n1m1-backward, blstm-n1m2-forward, blstm-n1m2-backward, blstm-n1m3-forward, blstm-n1m3-backward, blstm-n1m4-forward, blstm-n1m4-backward, blstm-n1m5-forward, blstm-n1m5-backward, blstm-n1m6-forward, blstm-n1m6-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm-n1m7-backward input=Append(blstm-n1m1-forward, blstm-n1m1-backward, blstm-n1m2-forward, blstm-n1m2-backward, blstm-n1m3-forward, blstm-n1m3-backward, blstm-n1m4-forward, blstm-n1m4-backward, blstm-n1m5-forward, blstm-n1m5-backward, blstm-n1m6-forward, blstm-n1m6-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=3 $lstm_opts

  fast-lstmp-layer name=blstm-n1t-forward input=Append(blstm-n1m1-forward, blstm-n1m1-backward, blstm-n1m2-forward, blstm-n1m2-backward, blstm-n1m3-forward, blstm-n1m3-backward, blstm-n1m4-forward, blstm-n1m4-backward, blstm-n1m5-forward, blstm-n1m5-backward, blstm-n1m6-forward, blstm-n1m6-backward, blstm-n1m7-forward, blstm-n1m7-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm-n1t-backward input=Append(blstm-n1m1-forward, blstm-n1m1-backward, blstm-n1m2-forward, blstm-n1m2-backward, blstm-n1m3-forward, blstm-n1m3-backward, blstm-n1m4-forward, blstm-n1m4-backward, blstm-n1m5-forward, blstm-n1m5-backward, blstm-n1m6-forward, blstm-n1m6-backward, blstm-n1m7-forward, blstm-n1m7-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=3 $lstm_opts

  fast-lstmp-layer name=blstm-n2m1-forward input=Append(blstm-n1t-forward, blstm-n1t-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm-n2m1-backward input=Append(blstm-n1t-forward, blstm-n1t-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=3 $lstm_opts
  fast-lstmp-layer name=blstm-n2m2-forward input=Append(blstm-n2m1-forward, blstm-n2m1-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm-n2m2-backward input=Append(blstm-n2m1-forward, blstm-n2m1-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=3 $lstm_opts
  fast-lstmp-layer name=blstm-n2m3-forward input=Append(blstm-n2m1-forward, blstm-n2m1-backward, blstm-n2m2-forward, blstm-n2m2-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm-n2m3-backward input=Append(blstm-n2m1-forward, blstm-n2m1-backward, blstm-n2m2-forward, blstm-n2m2-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=3 $lstm_opts
  fast-lstmp-layer name=blstm-n2m4-forward input=Append(blstm-n2m1-forward, blstm-n2m1-backward, blstm-n2m2-forward, blstm-n2m2-backward, blstm-n2m3-forward, blstm-n2m3-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm-n2m4-backward input=Append(blstm-n2m1-forward, blstm-n2m1-backward, blstm-n2m2-forward, blstm-n2m2-backward, blstm-n2m3-forward, blstm-n2m3-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=3 $lstm_opts
  fast-lstmp-layer name=blstm-n2m5-forward input=Append(blstm-n2m1-forward, blstm-n2m1-backward, blstm-n2m2-forward, blstm-n2m2-backward, blstm-n2m3-forward, blstm-n2m3-backward, blstm-n2m4-forward, blstm-n2m4-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm-n2m5-backward input=Append(blstm-n2m1-forward, blstm-n2m1-backward, blstm-n2m2-forward, blstm-n2m2-backward, blstm-n2m3-forward, blstm-n2m3-backward, blstm-n2m4-forward, blstm-n2m4-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=3 $lstm_opts
  fast-lstmp-layer name=blstm-n2m6-forward input=Append(blstm-n2m1-forward, blstm-n2m1-backward, blstm-n2m2-forward, blstm-n2m2-backward, blstm-n2m3-forward, blstm-n2m3-backward, blstm-n2m4-forward, blstm-n2m4-backward, blstm-n2m5-forward, blstm-n2m5-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm-n2m6-backward input=Append(blstm-n2m1-forward, blstm-n2m1-backward, blstm-n2m2-forward, blstm-n2m2-backward, blstm-n2m3-forward, blstm-n2m3-backward, blstm-n2m4-forward, blstm-n2m4-backward, blstm-n2m5-forward, blstm-n2m5-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=3 $lstm_opts
  fast-lstmp-layer name=blstm-n2m7-forward input=Append(blstm-n2m1-forward, blstm-n2m1-backward, blstm-n2m2-forward, blstm-n2m2-backward, blstm-n2m3-forward, blstm-n2m3-backward, blstm-n2m4-forward, blstm-n2m4-backward, blstm-n2m5-forward, blstm-n2m5-backward, blstm-n2m6-forward, blstm-n2m6-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm-n2m7-backward input=Append(blstm-n2m1-forward, blstm-n2m1-backward, blstm-n2m2-forward, blstm-n2m2-backward, blstm-n2m3-forward, blstm-n2m3-backward, blstm-n2m4-forward, blstm-n2m4-backward, blstm-n2m5-forward, blstm-n2m5-backward, blstm-n2m6-forward, blstm-n2m6-backward) cell-dim=$lstm_dim1 recurrent-projection-dim=$lstm_dim2 non-recurrent-projection-dim=$lstm_dim2 delay=3 $lstm_opts

  ## adding the layers for chain branch
  output-layer name=output input=Append(blstm-n2m1-forward, blstm-n2m1-backward, blstm-n2m2-forward, blstm-n2m2-backward, blstm-n2m3-forward, blstm-n2m3-backward, blstm-n2m4-forward, blstm-n2m4-backward, blstm-n2m5-forward, blstm-n2m5-backward, blstm-n2m6-forward, blstm-n2m6-backward, blstm-n2m7-forward, blstm-n2m7-backward) output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5 $output_opts

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=Append(blstm-n2m1-forward, blstm-n2m1-backward, blstm-n2m2-forward, blstm-n2m2-backward, blstm-n2m3-forward, blstm-n2m3-backward, blstm-n2m4-forward, blstm-n2m4-backward, blstm-n2m5-forward, blstm-n2m5-backward, blstm-n2m6-forward, blstm-n2m6-backward, blstm-n2m7-forward, blstm-n2m7-backward) output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5 $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 14 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/mini_librispeech-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=20 \
    --trainer.frames-per-iter=3000000 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=1 \
    --trainer.optimization.initial-effective-lrate=0.002 \
    --trainer.optimization.final-effective-lrate=0.0002 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --egs.chunk-width=$chunk_width \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 15 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test_tgsmall \
    $tree_dir $tree_dir/graph_tgsmall || exit 1;
fi

if [ $stage -le 16 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $tree_dir/graph_tgsmall data/${data}_hires ${dir}/decode_tgsmall_${data} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_{tgsmall,tglarge} \
       data/${data}_hires ${dir}/decode_{tgsmall,tglarge}_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

# Not testing the 'looped' decoding separately, because for
# TDNN systems it would give exactly the same results as the
# normal decoding.

if $test_online_decoding && [ $stage -le 17 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet3${nnet3_affix}/extractor ${dir} ${dir}_online

  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      # note: we just give it "data/${data}" as it only uses the wav.scp, the
      # feature type does not matter.
      steps/online/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --nj $nspk --cmd "$decode_cmd" \
        $tree_dir/graph_tgsmall data/${data} ${dir}_online/decode_tgsmall_${data} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_{tgsmall,tglarge} \
       data/${data}_hires ${dir}_online/decode_{tgsmall,tglarge}_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi


exit 0;
