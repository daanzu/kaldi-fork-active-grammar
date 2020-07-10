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

#define DEFAULT_VERBOSITY 0

namespace dragonfly {

using namespace kaldi;
using namespace fst;


struct AgfNNet3OnlineModelConfig : public BaseNNet3OnlineModelConfig {

    int32 nonterm_phones_offset = -1;  // offset from start of phones that start of nonterms are
    int32 rules_phones_offset = -1;  // offset from start of phones that the dictation nonterms are
    int32 dictation_phones_offset = -1;  // offset from start of phones that the kaldi_rules nonterms are
    std::string top_fst_filename;
    std::string dictation_fst_filename;

    bool Set(const std::string& name, const nlohmann::json& value) {
        if (BaseNNet3OnlineModelConfig::Set(name, value)) { return true; }
        if (name == "nonterm_phones_offset") { nonterm_phones_offset = value.get<int32>(); return true; }
        if (name == "rules_phones_offset") { rules_phones_offset = value.get<int32>(); return true; }
        if (name == "dictation_phones_offset") { dictation_phones_offset = value.get<int32>(); return true; }
        if (name == "top_fst_filename") { top_fst_filename = value.get<std::string>(); return true; }
        if (name == "dictation_fst_filename") { dictation_fst_filename = value.get<std::string>(); return true; }
        return false;
    }

    std::string ToString() {
        stringstream ss;
        ss << BaseNNet3OnlineModelConfig::ToString() << '\n';
        ss << "AgfNNet3OnlineModelConfig...";
        ss << "\n    " << "nonterm_phones_offset: " << nonterm_phones_offset;
        ss << "\n    " << "rules_phones_offset: " << rules_phones_offset;
        ss << "\n    " << "dictation_phones_offset: " << dictation_phones_offset;
        ss << "\n    " << "top_fst_filename: " << top_fst_filename;
        ss << "\n    " << "dictation_fst_filename: " << dictation_fst_filename;
        return ss.str();
    }
};

class AgfNNet3OnlineModelWrapper : public BaseNNet3OnlineModelWrapper {
    public:

        AgfNNet3OnlineModelWrapper(const std::string& model_dir, const std::string& config_str = "", int32 verbosity = DEFAULT_VERBOSITY);
        ~AgfNNet3OnlineModelWrapper();

        int32 AddGrammarFst(std::string& grammar_fst_filename);
        bool ReloadGrammarFst(int32 grammar_fst_index, std::string& grammar_fst_filename);
        bool RemoveGrammarFst(int32 grammar_fst_index);

        bool Decode(BaseFloat samp_freq, const Vector<BaseFloat>& frames, bool finalize, std::vector<bool>& grammars_activity, bool save_adaptation_state = true);
        void GetDecodedString(std::string& decoded_string, float* likelihood, float* am_score, float* lm_score, float* confidence, float* expected_error_rate) override;
        bool GetWordAlignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths, bool include_eps) override;

    protected:

        AgfNNet3OnlineModelConfig config_;

        // Model
        StdConstFst *top_fst_ = nullptr;
        StdConstFst *dictation_fst_ = nullptr;
        std::vector<StdConstFst*> grammar_fsts_;
        std::map<StdConstFst*, std::string> grammar_fsts_filename_map_;  // maps grammar_fst -> name; for debugging
        // INVARIANT: same size: grammar_fsts_, grammar_fsts_filename_map_
        std::vector<bool> grammars_activity_;

        // Model objects
        ActiveGrammarFst* active_grammar_fst_ = nullptr;

        // Decoder objects
        SingleUtteranceNnet3DecoderTpl<fst::ActiveGrammarFst>* decoder_ = nullptr;  // reinstantiated per utterance
        CombineRuleNontermMapper<CompactLatticeArc>* rule_relabel_mapper_ = nullptr;

        void StartDecoding() override;
        void CleanupDecoder() override;
};

}  // namespace dragonfly
