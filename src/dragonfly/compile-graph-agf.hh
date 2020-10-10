// compile-graph-agf

// Copyright   2019  David Zurow

// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License
// for more details.

// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "fstext/fstext-lib.h"
#include "fstext/push-special.h"
#include "fstext/grammar-context-fst.h"
#include "decoder/active-grammar-fst.h"
#include "fst/script/compile.h"

#include "nlohmann_json.hpp"

namespace dragonfly {

using namespace kaldi;
using namespace fst;


struct AgfCompilerConfig {
    std::string tree_rxfilename;
    std::string model_rxfilename;
    std::string lex_rxfilename;
    std::string grammar_rxfilename;
    std::string hclg_wxfilename;

    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 1.0;  // Caution: the script default is 0.1.
    int32 nonterm_phones_offset = -1;
    std::string disambig_rxfilename;
    bool verbose = false;

    bool compile_grammar = false;
    std::string grammar_symbols;
    bool topsort_grammar = false;
    bool arcsort_grammar = false;
    std::string grammar_prepend_nonterm_fst;
    std::string grammar_append_nonterm_fst;
    int32 grammar_prepend_nonterm = -1;
    int32 grammar_append_nonterm = -1;
    bool simplify_lg = true;  // Bool whether to simplify LG (do for command grammars, but not for dictation graph!)

    std::string word_syms_filename;
};

void from_json(const nlohmann::json& j, AgfCompilerConfig& c) {
    if (!j.is_object()) KALDI_ERR << "Not an object!";
    if (j.contains("tree_rxfilename")) j.at("tree_rxfilename").get_to(c.tree_rxfilename);
    if (j.contains("model_rxfilename")) j.at("model_rxfilename").get_to(c.model_rxfilename);
    if (j.contains("lex_rxfilename")) j.at("lex_rxfilename").get_to(c.lex_rxfilename);
    if (j.contains("grammar_rxfilename")) j.at("grammar_rxfilename").get_to(c.grammar_rxfilename);
    if (j.contains("hclg_wxfilename")) j.at("hclg_wxfilename").get_to(c.hclg_wxfilename);
    if (j.contains("transition_scale")) j.at("transition_scale").get_to(c.transition_scale);
    if (j.contains("self_loop_scale")) j.at("self_loop_scale").get_to(c.self_loop_scale);
    if (j.contains("nonterm_phones_offset")) j.at("nonterm_phones_offset").get_to(c.nonterm_phones_offset);
    if (j.contains("disambig_rxfilename")) j.at("disambig_rxfilename").get_to(c.disambig_rxfilename);
    if (j.contains("verbose")) j.at("verbose").get_to(c.verbose);
    if (j.contains("compile_grammar")) j.at("compile_grammar").get_to(c.compile_grammar);
    if (j.contains("grammar_symbols")) j.at("grammar_symbols").get_to(c.grammar_symbols);
    if (j.contains("topsort_grammar")) j.at("topsort_grammar").get_to(c.topsort_grammar);
    if (j.contains("arcsort_grammar")) j.at("arcsort_grammar").get_to(c.arcsort_grammar);
    if (j.contains("grammar_prepend_nonterm_fst")) j.at("grammar_prepend_nonterm_fst").get_to(c.grammar_prepend_nonterm_fst);
    if (j.contains("grammar_append_nonterm_fst")) j.at("grammar_append_nonterm_fst").get_to(c.grammar_append_nonterm_fst);
    if (j.contains("grammar_prepend_nonterm")) j.at("grammar_prepend_nonterm").get_to(c.grammar_prepend_nonterm);
    if (j.contains("grammar_append_nonterm")) j.at("grammar_append_nonterm").get_to(c.grammar_append_nonterm);
    if (j.contains("simplify_lg")) j.at("simplify_lg").get_to(c.simplify_lg);
    if (j.contains("word_syms_filename")) j.at("word_syms_filename").get_to(c.word_syms_filename);
}


class AgfCompiler {
   public:
    AgfCompiler(const AgfCompilerConfig& config);
    ~AgfCompiler() { };

    StdVectorFst* CompileGrammar(const StdVectorFst* grammar_fst_in, const std::string& hclg_wxfilename = "");
    StdVectorFst* CompileFstText(std::istream& grammar_text);

   private:
    AgfCompilerConfig config_;

    ContextDependency ctx_dep;  // the tree.
    TransitionModel trans_model;
    VectorFst<StdArc> *lex_fst;
    std::vector<int32> disambig_syms;
    std::vector<int32> phone_syms;

    fst::SymbolTable *word_syms_ = nullptr;
};

AgfCompiler::AgfCompiler(const AgfCompilerConfig& config) : config_(config) {
    if (config_.compile_grammar
        || !config_.grammar_rxfilename.empty()
        || !config_.hclg_wxfilename.empty()
        ) KALDI_ERR << "illegal config";

    ReadKaldiObject(config_.tree_rxfilename, &ctx_dep);

    ReadKaldiObject(config_.model_rxfilename, &trans_model);

    lex_fst = fst::ReadFstKaldi(config_.lex_rxfilename);

    if (config_.disambig_rxfilename != "")
      if (!ReadIntegerVectorSimple(config_.disambig_rxfilename, &disambig_syms))
        KALDI_ERR << "Could not read disambiguation symbols from "
                  << config_.disambig_rxfilename;
    if (disambig_syms.empty())
      KALDI_WARN << "You supplied no disambiguation symbols; note, these are "
                 << "typically necessary when compiling graphs from FSTs (i.e. "
                 << "supply L_disambig.fst and the list of disambig syms with\n"
                 << "--read-disambig-syms)";

    phone_syms = trans_model.GetPhones();
    SortAndUniq(&disambig_syms);
    for (int32 i = 0; i < disambig_syms.size(); i++)
      if (std::binary_search(phone_syms.begin(), phone_syms.end(),
                             disambig_syms[i]))
        KALDI_ERR << "Disambiguation symbol " << disambig_syms[i]
                  << " is also a phone.";

    if (!config_.word_syms_filename.empty())
        if (!(word_syms_ = fst::SymbolTable::ReadText(config_.word_syms_filename)))
            KALDI_ERR << "Could not read symbol table from file " << config_.word_syms_filename;
}

StdVectorFst* AgfCompiler::CompileGrammar(const StdVectorFst* grammar_fst_in, const std::string& hclg_wxfilename) {
    KALDI_VLOG(1) << "Preparing G...";

    VectorFst<StdArc>* grammar_fst = new StdVectorFst(*grammar_fst_in);

    if (config_.arcsort_grammar) {
      fst::ArcSort(grammar_fst, fst::ILabelCompare<StdArc>());
    }

    if (!config_.grammar_prepend_nonterm_fst.empty()) {
      VectorFst<StdArc> *nonterm_fst = fst::ReadFstKaldi(config_.grammar_prepend_nonterm_fst);
      fst::Concat(*nonterm_fst, grammar_fst);
    }
    if (!config_.grammar_append_nonterm_fst.empty()) {
      VectorFst<StdArc> *nonterm_fst = fst::ReadFstKaldi(config_.grammar_append_nonterm_fst);
      fst::Concat(grammar_fst, *nonterm_fst);
    }
    if (config_.grammar_prepend_nonterm > 0) {
      VectorFst<StdArc> nonterm_fst;
      nonterm_fst.AddState();
      nonterm_fst.SetStart(0);
      nonterm_fst.AddState();
      nonterm_fst.SetFinal(1, 0.0);
      nonterm_fst.AddArc(0, StdArc(config_.grammar_prepend_nonterm, 0, 0.0, 1));
      fst::Concat(nonterm_fst, grammar_fst);
    }
    if (config_.grammar_append_nonterm > 0) {
      VectorFst<StdArc> nonterm_fst;
      nonterm_fst.AddState();
      nonterm_fst.SetStart(0);
      nonterm_fst.AddState();
      nonterm_fst.SetFinal(1, 0.0);
      nonterm_fst.AddArc(0, StdArc(config_.grammar_append_nonterm, 0, 0.0, 1));
      fst::Concat(grammar_fst, nonterm_fst);
    }

    if (config_.simplify_lg) {
      // I think this should speed later stages
      KALDI_VLOG(1) << "Determinizing G fst...";
      VectorFst<StdArc> tmp_fst;
      Determinize(*grammar_fst, &tmp_fst);
      *grammar_fst = tmp_fst;
    }

    KALDI_VLOG(1) << "Composing LG...";
    VectorFst<StdArc> lg_fst;
    TableCompose(*lex_fst, *grammar_fst, &lg_fst);

    if (config_.topsort_grammar) {
      bool acyclic = fst::TopSort(&lg_fst);
      if (!acyclic) {
        KALDI_ERR
            << "Topological sorting of state-level lattice failed (probably"
            << " your lexicon has empty words or your LM has epsilon cycles"
            << ").";
      }
    }

    if (config_.simplify_lg) {
      // Remove epsilons to ease Determinization (Caster text manipulation hanging bug)
      // We need to use full RmEpsilon, because RemoveEpsLocal is not sufficient
      KALDI_VLOG(1) << "RmEpsiloning LG fst...";
      RmEpsilon(&lg_fst);
      // Disambiguate LG to ease Determinization (caspark's hanging bug)
      KALDI_VLOG(1) << "Disambiguating LG fst...";
      VectorFst<StdArc> tmp_fst;
      Disambiguate(lg_fst, &tmp_fst);
      lg_fst = tmp_fst;
    }

    KALDI_VLOG(1) << "Determinizing LG fst...";
    DeterminizeStarInLog(&lg_fst, fst::kDelta);

    KALDI_VLOG(1) << "Preparing LG fst...";
    MinimizeEncoded(&lg_fst, fst::kDelta);

    fst::PushSpecial(&lg_fst, fst::kDelta);

    delete grammar_fst;

    VectorFst<StdArc> clg_fst;

    std::vector<std::vector<int32> > ilabels;

    int32 context_width = ctx_dep.ContextWidth(),
        central_position = ctx_dep.CentralPosition();

    KALDI_VLOG(1) << "Composing CLG fst...";
    if (config_.nonterm_phones_offset < 0) {
      // The normal case.
      ComposeContext(disambig_syms, context_width, central_position,
                     &lg_fst, &clg_fst, &ilabels);
    } else {
      // The grammar-FST case. See ../doc/grammar.dox for an intro.
      if (context_width != 2 || central_position != 1) {
        KALDI_ERR << "Grammar-fst graph creation only supports models with left-"
            "biphone context.  (--nonterm-phones-offset option was supplied).";
      }
      ComposeContextLeftBiphone(config_.nonterm_phones_offset,  disambig_syms,
                                lg_fst, &clg_fst, &ilabels);
    }
    lg_fst.DeleteStates();

    KALDI_VLOG(1) << "Constructing H fst...";
    HTransducerConfig h_cfg;
    h_cfg.transition_scale = config_.transition_scale;
    h_cfg.nonterm_phones_offset = config_.nonterm_phones_offset;
    std::vector<int32> disambig_syms_h; // disambiguation symbols on
                                        // input side of H.
    VectorFst<StdArc> *h_fst = GetHTransducer(ilabels,
                                              ctx_dep,
                                              trans_model,
                                              h_cfg,
                                              &disambig_syms_h);

    KALDI_VLOG(1) << "Composing HCLG fst...";
    // VectorFst<StdArc> hclg_fst;  // transition-id to word.
    VectorFst<StdArc> *hclg_fst_p = new VectorFst<StdArc>();  // transition-id to word.
    VectorFst<StdArc> &hclg_fst = *hclg_fst_p;  // transition-id to word.
    TableCompose(*h_fst, clg_fst, &hclg_fst);
    clg_fst.DeleteStates();
    delete h_fst;

    KALDI_ASSERT(hclg_fst.Start() != fst::kNoStateId);

    KALDI_VLOG(1) << "Preparing HCLG fst...";
    // Epsilon-removal and determinization combined. This will fail if not determinizable.
    DeterminizeStarInLog(&hclg_fst);

    if (!disambig_syms_h.empty()) {
      RemoveSomeInputSymbols(disambig_syms_h, &hclg_fst);
      RemoveEpsLocal(&hclg_fst);
    }

    // Encoded minimization.
    MinimizeEncoded(&hclg_fst);

    std::vector<int32> disambig;
    bool check_no_self_loops = true,
        reorder = true;
    AddSelfLoops(trans_model,
                 disambig,
                 config_.self_loop_scale,
                 reorder,
                 check_no_self_loops,
                 &hclg_fst);

    if (config_.nonterm_phones_offset >= 0)
      PrepareForActiveGrammarFst(config_.nonterm_phones_offset, &hclg_fst);

    if (!hclg_wxfilename.empty()) {
        fst::ConstFst<StdArc> const_hclg(hclg_fst);
        bool binary = true, write_binary_header = false;  // suppress the ^@B
        Output ko(hclg_wxfilename, binary, write_binary_header);
        fst::FstWriteOptions wopts(PrintableWxfilename(hclg_wxfilename));
        const_hclg.Write(ko.Stream(), wopts);
    }

    KALDI_LOG << "Wrote graph with " << hclg_fst.NumStates()
              << " states to " << hclg_wxfilename;
    return hclg_fst_p;
}

StdVectorFst* AgfCompiler::CompileFstText(std::istream& grammar_text) {
    // FIXME: fix build for linux and macos, and CI for windows, to include fst::script for CompileFstInternal
    auto grammar_fstclass = fst::script::CompileFstInternal(grammar_text, "<AddGrammarFst>", "vector", "standard",
        word_syms_, word_syms_, nullptr, false, false, false, false, false);
    auto grammar_fst = dynamic_cast<StdVectorFst*>(fst::Convert(*grammar_fstclass->GetFst<StdArc>(), "vector"));
    if (!grammar_fst) KALDI_ERR << "could not convert grammar Fst to StdVectorFst";
    return grammar_fst;
}


int CompileGraphAgfMain(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;


    const char *usage =
        "Creates HCLG decoding graph.  Similar to mkgraph.sh but done in code.\n"
        "\n"
        "Usage:   compile-graph-agf [options] <tree-in> <model-in> <lexicon-fst-in> "
        " <gammar-rspecifier> <hclg-wspecifier>\n"
        "e.g.: \n"
        " compile-train-graphs-fsts tree 1.mdl L_disambig.fst G.fst HCLG.fst\n";
    ParseOptions po(usage);


    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 1.0;  // Caution: the script default is 0.1.
    int32 nonterm_phones_offset = -1;
    std::string disambig_rxfilename;


    po.Register("read-disambig-syms", &disambig_rxfilename, "File containing "
                "list of disambiguation symbols in phone symbol table");
    po.Register("transition-scale", &transition_scale, "Scale of transition "
                "probabilities (excluding self-loops).");
    po.Register("self-loop-scale", &self_loop_scale, "Scale of self-loop vs. "
                "non-self-loop probability mass.  Caution: the default of "
                "mkgraph.sh is 0.1, but this defaults to 1.0.");
    po.Register("nonterm-phones-offset", &nonterm_phones_offset, "Integer "
                "value of symbol #nonterm_bos in phones.txt, if present. "
                "(Only relevant for grammar decoding).");

    bool compile_grammar = false;
    std::string grammar_symbols;
    bool topsort_grammar = false;
    bool arcsort_grammar = false;
    std::string grammar_prepend_nonterm_fst;
    std::string grammar_append_nonterm_fst;
    int32 grammar_prepend_nonterm = -1;
    int32 grammar_append_nonterm = -1;
    bool simplify_lg = true;
    po.Register("compile-grammar", &compile_grammar, "");
    po.Register("grammar-symbols", &grammar_symbols, "");
    po.Register("topsort-grammar", &topsort_grammar, "");
    po.Register("arcsort-grammar", &arcsort_grammar, "");
    po.Register("grammar-prepend-nonterm-fst", &grammar_prepend_nonterm_fst, "");
    po.Register("grammar-append-nonterm-fst", &grammar_append_nonterm_fst, "");
    po.Register("grammar-prepend-nonterm", &grammar_prepend_nonterm, "");
    po.Register("grammar-append-nonterm", &grammar_append_nonterm, "");
    po.Register("simplify-lg", &simplify_lg, "Bool whether to simplify LG (do for command grammars, but not for dictation graph!)");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_rxfilename = po.GetArg(1),
        model_rxfilename = po.GetArg(2),
        lex_rxfilename = po.GetArg(3),
        grammar_rxfilename = po.GetArg(4),
        hclg_wxfilename = po.GetArg(5);

    ContextDependency ctx_dep;  // the tree.
    ReadKaldiObject(tree_rxfilename, &ctx_dep);

    TransitionModel trans_model;
    ReadKaldiObject(model_rxfilename, &trans_model);

    VectorFst<StdArc> *lex_fst = fst::ReadFstKaldi(lex_rxfilename);

    KALDI_VLOG(1) << "Preparing G...";

    VectorFst<StdArc> *grammar_fst;
    if (compile_grammar) {
      KALDI_ERR << "compile-grammar not supported";
      // kaldi::Input ki;
      // ki.OpenTextMode(grammar_rxfilename);
      // std::unique_ptr<const SymbolTable> isyms, osyms, ssyms;
      // isyms.reset(SymbolTable::ReadText(grammar_symbols));
      // if (!isyms) return 1;
      // osyms.reset(SymbolTable::ReadText(grammar_symbols));
      // if (!osyms) return 1;
      // auto compiled_fst = fst::script::CompileFstInternal(
      //     ki.Stream(), grammar_rxfilename, "vector", "standard", isyms.get(),
      //     osyms.get(), ssyms.get(), false, false, false, false, false);
      // // grammar_fst = fst::CastOrConvertToVectorFst(compiled_fst);
      // // grammar_fst = fst::CastOrConvertToVectorFst(compiled_fst->GetFst<StdArc>());
      // // auto f = VectorFst<StdArc>(*compiled_fst->GetFst<StdArc>());
      // grammar_fst = new VectorFst<StdArc>(*compiled_fst->GetFst<StdArc>());
      // // grammar_fst = fst::Convert<StdArc>(compiled_fst->GetFst<StdArc>(), "vector");
      // // grammar_fst = compiled_fst->GetFst<StdArc>();
      // // auto compiled_vectorfst = fst::script::VectorFstClass(*compiled_fst);
      // // grammar_fst = compiled_vectorfst.;
    } else {
      grammar_fst = fst::ReadFstKaldi(grammar_rxfilename);
    }

    if (arcsort_grammar) {
      fst::ArcSort(grammar_fst, fst::ILabelCompare<StdArc>());
    }

    if (!grammar_prepend_nonterm_fst.empty()) {
      VectorFst<StdArc> *nonterm_fst = fst::ReadFstKaldi(grammar_prepend_nonterm_fst);
      fst::Concat(*nonterm_fst, grammar_fst);
    }
    if (!grammar_append_nonterm_fst.empty()) {
      VectorFst<StdArc> *nonterm_fst = fst::ReadFstKaldi(grammar_append_nonterm_fst);
      fst::Concat(grammar_fst, *nonterm_fst);
    }
    if (grammar_prepend_nonterm > 0) {
      VectorFst<StdArc> nonterm_fst;
      nonterm_fst.AddState();
      nonterm_fst.SetStart(0);
      nonterm_fst.AddState();
      nonterm_fst.SetFinal(1, 0.0);
      nonterm_fst.AddArc(0, StdArc(grammar_prepend_nonterm, 0, 0.0, 1));
      fst::Concat(nonterm_fst, grammar_fst);
    }
    if (grammar_append_nonterm > 0) {
      VectorFst<StdArc> nonterm_fst;
      nonterm_fst.AddState();
      nonterm_fst.SetStart(0);
      nonterm_fst.AddState();
      nonterm_fst.SetFinal(1, 0.0);
      nonterm_fst.AddArc(0, StdArc(grammar_append_nonterm, 0, 0.0, 1));
      fst::Concat(grammar_fst, nonterm_fst);
    }

    if (simplify_lg) {
      // I think this should speed later stages
      KALDI_VLOG(1) << "Determinizing G fst...";
      VectorFst<StdArc> tmp_fst;
      Determinize(*grammar_fst, &tmp_fst);
      *grammar_fst = tmp_fst;
    }

    std::vector<int32> disambig_syms;
    if (disambig_rxfilename != "")
      if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_syms))
        KALDI_ERR << "Could not read disambiguation symbols from "
                  << disambig_rxfilename;
    if (disambig_syms.empty())
      KALDI_WARN << "You supplied no disambiguation symbols; note, these are "
                 << "typically necessary when compiling graphs from FSTs (i.e. "
                 << "supply L_disambig.fst and the list of disambig syms with\n"
                 << "--read-disambig-syms)";

    const std::vector<int32> &phone_syms = trans_model.GetPhones();
    SortAndUniq(&disambig_syms);
    for (int32 i = 0; i < disambig_syms.size(); i++)
      if (std::binary_search(phone_syms.begin(), phone_syms.end(),
                             disambig_syms[i]))
        KALDI_ERR << "Disambiguation symbol " << disambig_syms[i]
                  << " is also a phone.";

    KALDI_VLOG(1) << "Composing LG...";
    VectorFst<StdArc> lg_fst;
    TableCompose(*lex_fst, *grammar_fst, &lg_fst);

    if (topsort_grammar) {
      bool acyclic = fst::TopSort(&lg_fst);
      if (!acyclic) {
        KALDI_ERR
            << "Topological sorting of state-level lattice failed (probably"
            << " your lexicon has empty words or your LM has epsilon cycles"
            << ").";
      }
    }

    if (simplify_lg) {
      // Remove epsilons to ease Determinization (Caster text manipulation hanging bug)
      // We need to use full RmEpsilon, because RemoveEpsLocal is not sufficient
      KALDI_VLOG(1) << "RmEpsiloning LG fst...";
      RmEpsilon(&lg_fst);
      // Disambiguate LG to ease Determinization (caspark's hanging bug)
      KALDI_VLOG(1) << "Disambiguating LG fst...";
      VectorFst<StdArc> tmp_fst;
      Disambiguate(lg_fst, &tmp_fst);
      lg_fst = tmp_fst;
    }

    KALDI_VLOG(1) << "Determinizing LG fst...";
    DeterminizeStarInLog(&lg_fst, fst::kDelta);

    KALDI_VLOG(1) << "Preparing LG fst...";
    MinimizeEncoded(&lg_fst, fst::kDelta);

    fst::PushSpecial(&lg_fst, fst::kDelta);

    delete grammar_fst;
    delete lex_fst;

    VectorFst<StdArc> clg_fst;

    std::vector<std::vector<int32> > ilabels;

    int32 context_width = ctx_dep.ContextWidth(),
        central_position = ctx_dep.CentralPosition();

    KALDI_VLOG(1) << "Composing CLG fst...";
    if (nonterm_phones_offset < 0) {
      // The normal case.
      ComposeContext(disambig_syms, context_width, central_position,
                     &lg_fst, &clg_fst, &ilabels);
    } else {
      // The grammar-FST case. See ../doc/grammar.dox for an intro.
      if (context_width != 2 || central_position != 1) {
        KALDI_ERR << "Grammar-fst graph creation only supports models with left-"
            "biphone context.  (--nonterm-phones-offset option was supplied).";
      }
      ComposeContextLeftBiphone(nonterm_phones_offset,  disambig_syms,
                                lg_fst, &clg_fst, &ilabels);
    }
    lg_fst.DeleteStates();

    KALDI_VLOG(1) << "Constructing H fst...";
    HTransducerConfig h_cfg;
    h_cfg.transition_scale = transition_scale;
    h_cfg.nonterm_phones_offset = nonterm_phones_offset;
    std::vector<int32> disambig_syms_h; // disambiguation symbols on
                                        // input side of H.
    VectorFst<StdArc> *h_fst = GetHTransducer(ilabels,
                                              ctx_dep,
                                              trans_model,
                                              h_cfg,
                                              &disambig_syms_h);

    KALDI_VLOG(1) << "Composing HCLG fst...";
    VectorFst<StdArc> hclg_fst;  // transition-id to word.
    TableCompose(*h_fst, clg_fst, &hclg_fst);
    clg_fst.DeleteStates();
    delete h_fst;

    KALDI_ASSERT(hclg_fst.Start() != fst::kNoStateId);

    KALDI_VLOG(1) << "Preparing HCLG fst...";
    // Epsilon-removal and determinization combined. This will fail if not determinizable.
    DeterminizeStarInLog(&hclg_fst);

    if (!disambig_syms_h.empty()) {
      RemoveSomeInputSymbols(disambig_syms_h, &hclg_fst);
      RemoveEpsLocal(&hclg_fst);
    }

    // Encoded minimization.
    MinimizeEncoded(&hclg_fst);

    std::vector<int32> disambig;
    bool check_no_self_loops = true,
        reorder = true;
    AddSelfLoops(trans_model,
                 disambig,
                 self_loop_scale,
                 reorder,
                 check_no_self_loops,
                 &hclg_fst);

    if (nonterm_phones_offset >= 0)
      PrepareForActiveGrammarFst(nonterm_phones_offset, &hclg_fst);

    {  // convert 'hclg' to ConstFst and write.
      fst::ConstFst<StdArc> const_hclg(hclg_fst);
      bool binary = true, write_binary_header = false;  // suppress the ^@B
      Output ko(hclg_wxfilename, binary, write_binary_header);
      fst::FstWriteOptions wopts(PrintableWxfilename(hclg_wxfilename));
      const_hclg.Write(ko.Stream(), wopts);
    }

    KALDI_LOG << "Wrote graph with " << hclg_fst.NumStates()
              << " states to " << hclg_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << "Exception in compile-graph-agf";
    std::cerr << e.what();
    return -1;
  }
}

} // namespace dragonfly
