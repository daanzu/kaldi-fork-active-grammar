
all:
	-rm -f arpa2fst
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = compile-graph-agf

OBJFILES =

ADDLIBS = ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../lm/kaldi-lm.a \
          ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a


TESTFILES =

include ../makefiles/default_rules.mk
