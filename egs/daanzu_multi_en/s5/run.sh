#!/bin/bash

# Copyright 2016  Allen Guo
#           2017  Xiaohui Zhang
# Apache 2.0

. ./cmd.sh
. ./path.sh

# paths to corpora (see below for example)
ami=
fisher=
librispeech=
swbd=
tedlium2=
wsj0=
wsj1=
eval2000=
rt03=

set -ev
# check for kaldi_lm
which get_word_map.pl > /dev/null
if [ $? -ne 0 ]; then
  echo "This recipe requires installation of tools/kaldi_lm. Please run extras/kaldi_lm.sh in tools/" && exit 1;
fi

# preset paths
case $(hostname -d) in
  clsp.jhu.edu)
    ami=/export/corpora4/ami/amicorpus
    fisher="/export/corpora3/LDC/LDC2004T19 /export/corpora3/LDC/LDC2005T19 \
      /export/corpora3/LDC/LDC2004S13 /export/corpora3/LDC/LDC2005S13"
    librispeech=/export/a15/vpanayotov/data
    swbd=/export/corpora3/LDC/LDC97S62
    tedlium2=/export/corpora5/TEDLIUM_release2
    wsj0=/export/corpora5/LDC/LDC93S6B
    wsj1=/export/corpora5/LDC/LDC94S13B
    eval2000="/export/corpora/LDC/LDC2002S09/hub5e_00 /export/corpora/LDC/LDC2002T43"
    rt03="/export/corpora/LDC/LDC2007S10"
    hub4_en_96="/export/corpora/LDC/LDC97T22/hub4_eng_train_trans /export/corpora/LDC/LDC97S44/data"
    hub4_en_97="/export/corpora/LDC/LDC98T28/hub4e97_trans_980217 /export/corpora/LDC/LDC98S71/97_eng_bns_hub4"
    ;;
  cs.columbia.edu)
    librispeech=corpora/librispeech
    tedlium3=corpora/TEDLIUM_release-3
    commonvoice=corpora/cv_corpus_v3
    tatoeba=corpora/tatoeba
    speechcommands=corpora/speechcommands
    ;;
esac

# general options
stage=1
cleanup_stage=1
multi=multi_a  # This defines the "variant" we're using; see README.md
srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"

. utils/parse_options.sh

# Prepare the basic dictionary (a combination of swbd+CMU+tedlium lexicons) in data/local/dict_combined.
# and train a G2P model using the combined lexicon in data/local/dict_combined
if false && [ $stage -le 1 ]; then
  # We prepare the basic dictionary in data/local/dict_combined.
  local/prepare_dict.sh $tedlium2
  (
   steps/dict/train_g2p_phonetisaurus.sh --stage 0 --silence-phones \
     "data/local/dict_combined/silence_phones.txt" data/local/dict_combined/lexicon.txt exp/g2p || touch exp/g2p/.error
  ) &
fi

if true && [ $stage -le 1 ]; then
  local/cmu_tedlium_prepare_dict.sh $tedlium3
  dir=data/local/dict_cmu_tedlium
  (
   steps/dict/train_g2p_phonetisaurus.sh --stage 0 --silence-phones \
     "${dir}/silence_phones.txt" ${dir}/lexicon.txt exp/g2p || touch exp/g2p/.error
  ) &
fi

if false && [ $stage -le 1 ]; then
  dir=data/local/dict_zamia
  mkdir -p $dir
  for w in SIL SPN NSN; do echo $w; done > $dir/silence_phones.txt
  echo SIL > $dir/optional_silence.txt
  [ -f $dir/lexicon.txt ] || exit 1
  [ -f $dir/nonsilence_phones.txt ] || exit 1
  [ -f $dir/extra_questions.txt ] || exit 1
  utils/validate_dict_dir.pl $dir

  (
   steps/dict/train_g2p_phonetisaurus.sh --stage 0 --silence-phones \
     "${dir}/silence_phones.txt" ${dir}/lexicon.txt exp/g2p || touch exp/g2p/.error
  ) &
fi

# wait && exit

dict_root=data/local/dict_cmu_tedlium

# Prepare corpora data
if [ $stage -le 2 ]; then
  mkdir -p data/local
  # librispeech
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    local/librispeech_download_and_untar.sh $librispeech www.openslr.org/resources/12 $part
  done
  local/librispeech_data_prep.sh $librispeech/LibriSpeech/train-clean-100 data/librispeech_100/train
  local/librispeech_data_prep.sh $librispeech/LibriSpeech/train-clean-360 data/librispeech_360/train
  local/librispeech_data_prep.sh $librispeech/LibriSpeech/train-other-500 data/librispeech_500/train
  local/librispeech_data_prep.sh $librispeech/LibriSpeech/test-clean data/librispeech/test
  local/librispeech_data_prep.sh $librispeech/LibriSpeech/test-other data/librispeech_other/test
  # tedlium
  local/tedlium_download_data.sh
  local/tedlium_prepare_data.sh $tedlium3
  # commonvoice
  local/commonvoice_download_and_untar.sh $commonvoice https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/en.tar.gz
  for part in train dev test validated; do  # other invalidated
    local/commonvoice_data_prep.py $commonvoice $part data/cv_$part
  done
  # tatoeba
  local/tatoeba_download_and_unzip.sh $tatoeba https://downloads.tatoeba.org/audio/tatoeba_audio_eng.zip
  local/tatoeba_data_prep.py $tatoeba data/tatoeba
  # speechcommands
  local/speechcommands_download_and_untar.sh $speechcommands http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
  local/speechcommands_data_prep.py $speechcommands/speech_commands_v0.02 data/speechcommands
fi

train_corpora="librispeech_100/train librispeech_360/train librispeech_500/train tedlium/train cv_train cv_dev cv_test cv_validated tatoeba speechcommands"
# test_corpora="librispeech/test tedlium/test"
test_corpora="librispeech/test"

# Normalize transcripts
if true && [ $stage -le 3 ]; then
  for f in data/*/{train,test}/text; do
    echo Normalizing $f
    cp $f $f.orig
    local/normalize_transcript.py $f.orig > $f
  done
fi

# Synthesize pronounciations for OOV words across all training transcripts and produce the final lexicon.
if true && [ $stage -le 4 ]; then
  wait # Waiting for train_g2p.sh to finish
  if [ -f exp/g2p/.error ]; then
     rm exp/g2p/.error || true
     echo "Fail to train the G2P model." && exit 1;
  fi
  dict_dir=${dict_root}_nosp
  mkdir -p $dict_dir
  rm $dict_dir/lexiconp.txt 2>/dev/null || true
  cp ${dict_root}/{extra_questions,nonsilence_phones,silence_phones,optional_silence}.txt $dict_dir

  echo 'Gathering missing words...'
  
  lexicon=${dict_root}/lexicon.txt
  g2p_tmp_dir=data/local/g2p_phonetisarus
  mkdir -p $g2p_tmp_dir

  # awk command from http://stackoverflow.com/questions/2626274/print-all-but-the-first-three-columns
  cat data/*/train/text | \
    local/count_oovs.pl $lexicon | \
    awk '{if (NF > 3 ) {for(i=4; i<NF; i++) printf "%s ",$i; print $NF;}}' | \
    perl -ape 's/\s/\n/g;' | \
    sort | uniq > $g2p_tmp_dir/missing.txt
  cat $g2p_tmp_dir/missing.txt | \
    grep "^[a-z]*$"  > $g2p_tmp_dir/missing_onlywords.txt

  echo "Missing words:"
  cat $g2p_tmp_dir/missing_onlywords.txt
  echo "Total $(cat $g2p_tmp_dir/missing_onlywords.txt | wc -l) in $g2p_tmp_dir/missing_onlywords.txt"

  steps/dict/apply_g2p_phonetisaurus.sh --nbest 1 $g2p_tmp_dir/missing_onlywords.txt exp/g2p exp/g2p/oov_lex || exit 1;
  cp exp/g2p/oov_lex/lexicon.lex $g2p_tmp_dir/missing_lexicon.txt

  extended_lexicon=$dict_dir/lexicon.txt
  echo "Adding new pronunciations to get extended lexicon $extended_lexicon"
  cat <(cut -f 1,3 $g2p_tmp_dir/missing_lexicon.txt) $lexicon | sort | uniq > $extended_lexicon
fi

# if true && [ $stage -le 4 ]; then
#   cp -rp ${dict_root} ${dict_root}_nosp
# fi

# We'll do multiple iterations of pron/sil-prob estimation. So the structure of
# the dict/lang dirs are designed as ${dict/lang_root}_${dict_affix}, where dict_affix
# is "nosp" or the name of the acoustic model we use to estimate pron/sil-probs.
# dict_root=data/local/dict
lang_root=data/lang

function do_prepare_lm ()
{
  # $@: dict_affix
  [ $# == 1 ] || (echo "do_prepare_lm: wrong number of arguments"; exit 1)
  local dict_affix=$1
  utils/prepare_lang.sh ${dict_root}_${dict_affix} "<unk>" data/local/tmp/lang_${dict_affix} ${lang_root}_${dict_affix}
  utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
    ${lang_root}_${dict_affix} \
    data/local/lm/enwiki-3.arpa.gz \
    ${lang_root}_${dict_affix}_daanzu_tg
}

# prepare (and validate) lang directory
if false && [ $stage -le 5 ]; then
  utils/prepare_lang.sh ${dict_root}_nosp "<unk>" data/local/tmp/lang_nosp ${lang_root}_nosp
fi

# prepare LM and test lang directory
if false && [ $stage -le 6 ]; then
  mkdir -p data/local/lm
  cat data/{fisher,swbd}/train/text > data/local/lm/text
  local/train_lms.sh  # creates data/local/lm/3gram-mincount/lm_unpruned.gz
  utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
    ${lang_root}_nosp data/local/lm/3gram-mincount/lm_unpruned.gz \
    ${dict_root}_nosp/lexicon.txt ${lang_root}_nosp_fsh_sw1_tg
fi

if true && [ $stage -le 6 ]; then
  # utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
  #   ${lang_root}_nosp data/local/lm/enwiki-3.arpa.gz \
  #   ${dict_root}_nosp/lexicon.txt ${lang_root}_nosp_daanzu_tg
  do_prepare_lm nosp
fi

# lm=${lang_root}_nosp_daanzu_tg
# [ -d ${lm} ] || echo "Expected LM in $lm" && exit 1

# make training features
if [ $stage -le 7 ]; then
  mfccdir=mfcc
  corpora=$train_corpora
  # path to 'train' directory
  for c in $corpora; do
    (
     data=data/$c
     steps/make_mfcc.sh --mfcc-config conf/mfcc.conf \
       --cmd "$train_cmd" --nj 40 \
       $data exp/make_mfcc/$c/train || touch $data/.error
     steps/compute_cmvn_stats.sh \
       $data exp/make_mfcc/$c/train || touch $data/.error
    ) &
  done
  wait
  if [ -f $data/.error ]; then
     rm $data/.error || true
     echo "Fail to extract features." && exit 1;
  fi
fi

# fix and validate training data directories
if [ $stage -le 8 ]; then
  # get rid of spk2gender files because not all corpora have them
  for c in $train_corpora; do
    rm data/$c/spk2gender 2>/dev/null || true
  done
  # create reco2channel_and_file files for wsj and librispeech
  for c in librispeech_100 librispeech_360 librispeech_500; do
    awk '{print $1, $1, "A"}' data/$c/train/wav.scp > data/$c/train/reco2file_and_channel;
  done
  # apply standard fixes, then validate
  for c in $train_corpora; do
    utils/fix_data_dir.sh data/$c
    utils/validate_data_dir.sh data/$c
  done
fi

# make test features
if [ $stage -le 9 ]; then
  mfccdir=mfcc
  corpora=$test_corpora
  # path to 'test' directory
  for c in $corpora; do
    data=data/$c
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf \
      --cmd "$train_cmd" --nj 20 \
      $data exp/make_mfcc/$c/test || exit 1;
    steps/compute_cmvn_stats.sh \
      $data exp/make_mfcc/$c/test || exit 1;
  done
fi

# fix and validate test data directories
if [ $stage -le 10 ]; then
  for c in $test_corpora; do
    utils/fix_data_dir.sh data/$c
    utils/validate_data_dir.sh data/$c
  done
fi

function do_decode_test_gmm ()
{
  # $@: gmm_name (mono), graph_name (graph_tg), lang_name (_nosp_daanzu_tg), decode_name (decode_tg)
  [ $# == 4 ] || (echo "do_decode_test_gmm: wrong number of arguments"; exit 1)
  local gmm=$1
  local graph_dir=exp/$multi/$gmm/$2
  utils/mkgraph.sh ${lang_root}${3} \
    exp/$multi/$gmm $graph_dir || exit 1;
  for e in $test_corpora; do
    steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" --config conf/decode.config $graph_dir \
      data/$e exp/$multi/$gmm/${4}_${e%/test} || exit 1;
  done
}

# train mono on swbd 10k short (nodup)
if [ $stage -le 11 ]; then
  local/make_partitions.sh --multi $multi --stage 1 || exit 1;
  steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
    data/$multi/mono ${lang_root}_nosp exp/$multi/mono || exit 1;
  do_decode_test_gmm mono graph_tg _nosp_daanzu_tg decode_tg &
fi

# train tri1a and tri1b (first and second triphone passes) on swbd 30k (nodup)
if [ $stage -le 12 ]; then
  local/make_partitions.sh --multi $multi --stage 2 || exit 1;
  steps/align_si.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
    data/$multi/mono_ali ${lang_root}_nosp exp/$multi/mono exp/$multi/mono_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 3200 30000 \
    data/$multi/tri1a ${lang_root}_nosp exp/$multi/mono_ali exp/$multi/tri1a || exit 1;

  steps/align_si.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
    data/$multi/tri1a_ali ${lang_root}_nosp exp/$multi/tri1a exp/$multi/tri1a_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 3200 30000 \
    data/$multi/tri1b ${lang_root}_nosp exp/$multi/tri1a_ali exp/$multi/tri1b || exit 1;
  do_decode_test_gmm tri1b graph_tg _nosp_daanzu_tg decode_tg &
fi

# train tri2 (third triphone pass) on swbd 100k (nodup)
if [ $stage -le 13 ]; then
 local/make_partitions.sh --multi $multi --stage 3 || exit 1;
 steps/align_si.sh --boost-silence 1.25 --nj 50 --cmd "$train_cmd" \
   data/$multi/tri1b_ali ${lang_root}_nosp exp/$multi/tri1b exp/$multi/tri1b_ali || exit 1;
 steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 5500 90000 \
   data/$multi/tri2 ${lang_root}_nosp exp/$multi/tri1b_ali exp/$multi/tri2 || exit 1;
fi

# train tri3a (4th triphone pass) on whole swbd
if [ $stage -le 14 ]; then
  local/make_partitions.sh --multi $multi --stage 4 || exit 1;
  steps/align_si.sh --boost-silence 1.25 --nj 100 --cmd "$train_cmd" \
    data/$multi/tri2_ali ${lang_root}_nosp exp/$multi/tri2 exp/$multi/tri2_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 11500 200000 \
    data/$multi/tri3a ${lang_root}_nosp exp/$multi/tri2_ali exp/$multi/tri3a || exit 1;
  do_decode_test_gmm tri3a graph_tg _nosp_daanzu_tg decode_tg &
fi

# train tri3b (LDA+MLLT) on whole fisher + swbd (nodup)
if [ $stage -le 15 ]; then
  local/make_partitions.sh --multi $multi --stage 5 || exit 1;
  steps/align_si.sh --boost-silence 1.25 --nj 100 --cmd "$train_cmd" \
    data/$multi/tri3a_ali ${lang_root}_nosp exp/$multi/tri3a exp/$multi/tri3a_ali || exit 1;
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 11500 400000 \
    data/$multi/tri3b ${lang_root}_nosp exp/$multi/tri3a_ali exp/$multi/tri3b || exit 1;
  do_decode_test_gmm tri3b graph_tg _nosp_daanzu_tg decode_tg &
fi

# reestimate pron & sil-probs
dict_affix=${multi}_tri3b
if [ $stage -le 16 ]; then
  gmm=tri3b
  steps/get_prons.sh --cmd "$train_cmd" data/$multi/$gmm ${lang_root}_nosp exp/$multi/$gmm
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    ${dict_root}_nosp exp/$multi/$gmm/pron_counts_nowb.txt \
    exp/$multi/$gmm/sil_counts_nowb.txt exp/$multi/$gmm/pron_bigram_counts_nowb.txt ${dict_root}_${dict_affix}
  do_prepare_lm $dict_affix
  do_decode_test_gmm $gmm graph_tg_sp _${dict_affix}_daanzu_tg decode_tg_sp &
fi

lang=${lang_root}_${dict_affix}
if false && [ $stage -le 17 ]; then
  # This does the actual data cleanup.
  steps/cleanup/clean_and_segment_data.sh --stage $cleanup_stage --nj 100 --cmd "$train_cmd" \
    data/tedlium/train $lang exp/$multi/tri3b exp/$multi/tri3b_tedlium_cleaning_work data/$multi/tedlium_cleaned/train
fi

# train tri4 on fisher + swbd + tedlium (nodup)
if [ $stage -le 18 ]; then
  local/make_partitions.sh --multi $multi --stage 6 || exit 1;
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 100 \
    data/$multi/tri3b_ali $lang \
    exp/$multi/tri3b exp/$multi/tri3b_ali || exit 1;
  steps/train_sat.sh --cmd "$train_cmd" 11500 800000 \
    data/$multi/tri4 $lang exp/$multi/tri3b_ali exp/$multi/tri4 || exit 1;
  do_decode_test_gmm tri4 graph_tg _${dict_affix}_daanzu_tg decode_tg &
fi

# train tri5a on fisher + swbd + tedlium + wsj + hub4_en (nodup)
if [ $stage -le 19 ]; then
  local/make_partitions.sh --multi $multi --stage 7 || exit 1;
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 100 \
    data/$multi/tri4_ali $lang \
    exp/$multi/tri4 exp/$multi/tri4_ali || exit 1;
  steps/train_sat.sh --cmd "$train_cmd" 11500 1600000 \
    data/$multi/tri5a $lang exp/$multi/tri4_ali exp/$multi/tri5a || exit 1;
  do_decode_test_gmm tri5a graph_tg _${dict_affix}_daanzu_tg decode_tg &
fi

# reestimate pron & sil-probs
dict_affix=${multi}_tri5a
if [ $stage -le 20 ]; then
  gmm=tri5a
  steps/get_prons.sh --cmd "$train_cmd" data/$multi/$gmm ${lang_root}_nosp exp/$multi/$gmm
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    ${dict_root}_nosp exp/$multi/$gmm/pron_counts_nowb.txt \
    exp/$multi/$gmm/sil_counts_nowb.txt exp/$multi/$gmm/pron_bigram_counts_nowb.txt ${dict_root}_${dict_affix}
  do_prepare_lm $dict_affix
  do_decode_test_gmm tri5a graph_tg_sp _${dict_affix}_daanzu_tg decode_tg_sp &
fi

lang=${lang_root}_${dict_affix}
# train tri5b on fisher + swbd + tedlium + wsj + hub4_en + librispeeh460 (nodup)
if [ $stage -le 21 ]; then
  local/make_partitions.sh --multi $multi --stage 8 || exit 1;
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 100 \
    data/$multi/tri5a_ali $lang \
    exp/$multi/tri5a exp/$multi/tri5a_ali || exit 1;
  steps/train_sat.sh --cmd "$train_cmd" 11500 2000000 \
    data/$multi/tri5b $lang exp/$multi/tri5a_ali exp/$multi/tri5b || exit 1;
  do_decode_test_gmm tri5b graph_tg _${dict_affix}_daanzu_tg decode_tg &
fi

# train tri6a on fisher + swbd + tedlium + wsj + hub4_en + librispeeh960 (nodup)
if [ $stage -le 22 ]; then
  local/make_partitions.sh --multi $multi --stage 9 || exit 1;
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 100 \
    data/$multi/tri5b_ali $lang \
    exp/$multi/tri5b exp/$multi/tri5b_ali || exit 1;

  steps/train_sat.sh --cmd "$train_cmd" 14000 2400000 \
    data/$multi/tri6a $lang exp/$multi/tri5b_ali exp/$multi/tri6a || exit 1;
  do_decode_test_gmm tri6a graph_tg _${dict_affix}_daanzu_tg decode_tg &
fi
# wait; exit 0  ######################################################################################################################################################
