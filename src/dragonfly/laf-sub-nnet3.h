// NNet3 AGF

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

#include "feat/wave-reader.h"
#include "online2/online-feature-pipeline.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/confidence.h"
#include "lat/lattice-functions.h"
#include "lat/sausages.h"
#include "lat/word-align-lattice-lexicon.h"
#include "nnet3/nnet-utils.h"
#include "decoder/active-grammar-fst.h"

#include "base-nnet3.h"
#include "utils.h"
#include "kaldi-utils.h"
#include "nlohmann_json.hpp"

namespace dragonfly {

using namespace kaldi;
using namespace fst;


struct LafNNet3OnlineModelConfig : public BaseNNet3OnlineModelConfig {
    using Ptr = std::shared_ptr<LafNNet3OnlineModelConfig>;

    static constexpr auto Create = BaseNNet3OnlineModelConfig::Create<LafNNet3OnlineModelConfig>;

    std::string hcl_fst_filename = "HCLr.fst";
    std::string disambig_tids_filename = "disambig_tids.int";
    std::string relabel_ilabels_filename;
    std::string word_syms_relabeled_filename;
    std::string dictation_fst_filename;
    int32 rules_words_offset = 1000000;
    int32 max_num_rules = 9999;
    size_t decode_fst_cache_size = 1ULL << 30;  // Note: this is used independently for 3 separate Fsts! FIXME: should we adjust this based on size of grammars + dictation fsts?

    bool Set(const std::string& name, const nlohmann::json& value) override {
        if (BaseNNet3OnlineModelConfig::Set(name, value)) { return true; }
        if (name == "hcl_fst_filename") { value.get_to(hcl_fst_filename); return true; }
        if (name == "disambig_tids_filename") { value.get_to(disambig_tids_filename); return true; }
        if (name == "relabel_ilabels_filename") { value.get_to(relabel_ilabels_filename); return true; }
        if (name == "word_syms_relabeled_filename") { value.get_to(word_syms_relabeled_filename); return true; }
        if (name == "dictation_fst_filename") { value.get_to(dictation_fst_filename); return true; }
        if (name == "rules_words_offset") { value.get_to(rules_words_offset); return true; }
        if (name == "max_num_rules") { value.get_to(max_num_rules); return true; }
        if (name == "decode_fst_cache_size") { value.get_to(decode_fst_cache_size); return true; }
        return false;
    }

    std::string ToString() override {
        stringstream ss;
        ss << BaseNNet3OnlineModelConfig::ToString() << '\n';
        ss << "LafNNet3OnlineModelConfig...";
        ss << "\n    " << "hcl_fst_filename: " << hcl_fst_filename;
        ss << "\n    " << "disambig_tids_filename: " << disambig_tids_filename;
        ss << "\n    " << "relabel_ilabels_filename: " << relabel_ilabels_filename;
        ss << "\n    " << "word_syms_relabeled_filename: " << word_syms_relabeled_filename;
        ss << "\n    " << "dictation_fst_filename: " << dictation_fst_filename;
        ss << "\n    " << "rules_words_offset: " << rules_words_offset;
        ss << "\n    " << "max_num_rules: " << max_num_rules;
        ss << "\n    " << "decode_fst_cache_size: " << decode_fst_cache_size;
        return ss.str();
    }
};

class LafNNet3OnlineModelWrapper : public BaseNNet3OnlineModelWrapper {
    public:

        LafNNet3OnlineModelWrapper(LafNNet3OnlineModelConfig::Ptr config, int32 verbosity = DEFAULT_VERBOSITY);
        ~LafNNet3OnlineModelWrapper() override;

        void PrepareGrammarFst(fst::StdVectorFst* grammar_fst, bool relabel);
        int32 AddGrammarFst(fst::StdExpandedFst* grammar_fst, std::string grammar_name = "<unnamed>");  // Does not take ownership of FST!
        int32 AddGrammarFst(std::istream& grammar_text);
        int32 AddGrammarFst(std::string& grammar_fst_filename);
        bool ReloadGrammarFst(int32 grammar_fst_index, fst::StdExpandedFst* grammar_fst, std::string grammar_name = "<unnamed>");  // Does not take ownership of FST!
        bool RemoveGrammarFst(int32 grammar_fst_index);
        void SetActiveGrammars(const std::vector<bool>& grammars_activity) { grammars_activity_ = grammars_activity; };

        bool Decode(BaseFloat samp_freq, const Vector<BaseFloat>& frames, bool finalize, const std::vector<bool>& grammars_activity, bool save_adaptation_state = true);
        bool Decode(BaseFloat samp_freq, const Vector<BaseFloat>& frames, bool finalize, bool save_adaptation_state = true) override;
        void GetDecodedString(std::string& decoded_string, float* likelihood, float* am_score, float* lm_score, float* confidence, float* expected_error_rate) override;

    protected:

        LafNNet3OnlineModelConfig::Ptr config_;

        // Model
        StdFst *hcl_fst_ = nullptr;
        std::vector<int32> disambig_tids_;
        std::vector<std::pair<StdArc::Label, StdArc::Label>> relabel_ilabels_;  // Lookahead relabel mapping (word-ids -> relabeled-word-ids)
        fst::SymbolTable *word_syms_relabeled_ = nullptr;  // Word symbol table composed with relabeling, allowing compiling directly to relabeled grammar
        StdConstFst *dictation_fst_ = nullptr;
        std::vector<StdExpandedFst*> grammar_fsts_;
        std::map<StdFst*, std::string> grammar_fsts_name_map_;  // maps grammar_fst -> name; for debugging
        // INVARIANT: same size: grammar_fsts_, grammar_fsts_name_map_
        std::vector<bool> grammars_activity_;  // bitfield of whether each grammar is active for current/upcoming utterance

        // Model objects
        StdFst* decode_fst_ = nullptr;
        std::vector<bool> decode_fst_grammars_activity_;  // grammars_activity_ for decode_fst_ creation

        // Decoder objects
        SingleUtteranceNnet3DecoderTpl<fst::StdFst>* decoder_ = nullptr;  // reinstantiated per utterance
        CombineRuleNontermMapper<CompactLatticeArc>* rule_relabel_mapper_ = nullptr;

        void BuildDecodeFst();
        void DestroyDecodeFst();
        void StartDecoding() override;
        void CleanupDecoder() override;
};

} // namespace dragonfly
