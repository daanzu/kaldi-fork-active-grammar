# adapted from decoder/Makefile

all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

TESTFILES =

OBJFILES = active-grammar-fst.o agf-nnet3.o

LIBNAME = kaldi-dragonfly

ADDLIBS = ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
