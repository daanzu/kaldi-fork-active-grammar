# adapted from decoder/Makefile

all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = compile-graph-agf

TESTFILES =

OBJFILES = agf-nnet3.o

LIBNAME = kaldi-dragonfly

ADDLIBS = ../ivector/kaldi-ivector.a ../nnet3/kaldi-nnet3.a \
          ../chain/kaldi-chain.a ../nnet2/kaldi-nnet2.a \
          ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a ../feat/kaldi-feat.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a ../online2/kaldi-online2.a ../fstext/kaldi-fstext.a 

include ../makefiles/default_rules.mk
