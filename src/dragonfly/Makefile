# adapted from decoder/Makefile

all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk
EXTRA_LDLIBS = $(subst libfst,libfstscript,$(OPENFSTLIBS))

TESTFILES =

OBJFILES = base-nnet3.o agf-sub-nnet3.o plain-sub-nnet3.o laf-sub-nnet3.o fst-export.o md5.o

LIBNAME = kaldi-dragonfly
DynamicLibrary = kaldi-dragonfly
DynamicLibraryExtras = dragonfly.h windows-targetver.h windows-dllmain.cc

ADDLIBS = ../ivector/kaldi-ivector.a ../nnet3/kaldi-nnet3.a \
          ../chain/kaldi-chain.a ../nnet2/kaldi-nnet2.a \
          ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a ../feat/kaldi-feat.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a ../online2/kaldi-online2.a ../fstext/kaldi-fstext.a \
          ../lm/kaldi-lm.a ../rnnlm/kaldi-rnnlm.a

include ../makefiles/default_rules.mk
