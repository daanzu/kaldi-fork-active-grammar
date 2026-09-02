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

#include <memory>

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

#include "active-base-nnet3.h"
#include "utils.h"
#include "kaldi-utils.h"
#include "nlohmann_json.hpp"

namespace dragonfly {

using namespace kaldi;
using namespace fst;


struct AgfNNet3OnlineModelConfig : public ActiveBaseNNet3OnlineModelConfig {
    using Ptr = std::shared_ptr<AgfNNet3OnlineModelConfig>;

    static constexpr auto Create = BaseNNet3OnlineModelConfig::Create<AgfNNet3OnlineModelConfig>;

    int32 nonterm_phones_offset = -1;  // offset from start of phones that start of nonterms are
    int32 rules_phones_offset = -1;  // offset from start of phones that the kaldi_rules nonterms are
    int32 dictation_phones_offset = -1;  // offset from start of phones that the dictation nonterms are
    uint64 top_fst = 0;  // actually a void* pointer to the top FST object
    std::string top_fst_filename;

    bool Set(const std::string& name, const nlohmann::json& value) override {
        if (ActiveBaseNNet3OnlineModelConfig::Set(name, value)) { return true; }
        if (name == "nonterm_phones_offset") { value.get_to(nonterm_phones_offset); return true; }
        if (name == "rules_phones_offset") { value.get_to(rules_phones_offset); return true; }
        if (name == "dictation_phones_offset") { value.get_to(dictation_phones_offset); return true; }
        if (name == "top_fst") { value.get_to(top_fst); return true; }
        if (name == "top_fst_filename") { value.get_to(top_fst_filename); return true; }
        return false;
    }

    std::string ToString() override {
        stringstream ss;
        ss << ActiveBaseNNet3OnlineModelConfig::ToString() << '\n';
        ss << "AgfNNet3OnlineModelConfig...";
        ss << "\n    " << "nonterm_phones_offset: " << nonterm_phones_offset;
        ss << "\n    " << "rules_phones_offset: " << rules_phones_offset;
        ss << "\n    " << "dictation_phones_offset: " << dictation_phones_offset;
        ss << "\n    " << "top_fst: " << top_fst;
        ss << "\n    " << "top_fst_filename: " << top_fst_filename;
        return ss.str();
    }
};

class AgfNNet3OnlineModelWrapper : public ActiveBaseNNet3OnlineModelWrapper {
    public:

        AgfNNet3OnlineModelWrapper(AgfNNet3OnlineModelConfig::Ptr config, int32 verbosity = DEFAULT_VERBOSITY);
        ~AgfNNet3OnlineModelWrapper() override;

        int32 AddGrammarFst(int32 grammar_fst_index, std::unique_ptr<fst::StdConstFst> grammar_fst, std::string grammar_name = "<unnamed>");
        int32 AddGrammarFst(int32 grammar_fst_index, std::string& grammar_fst_filename);
        bool ReloadGrammarFst(int32 grammar_fst_index, std::unique_ptr<fst::StdConstFst> grammar_fst, std::string grammar_name = "<unnamed>");
        bool ReloadGrammarFst(int32 grammar_fst_index, std::string& grammar_fst_filename);
        bool RemoveGrammarFst(int32 grammar_fst_index);

        bool Decode(BaseFloat samp_freq, const Vector<BaseFloat>& frames, bool finalize, bool save_adaptation_state = true) override;
        void GetDecodedString(std::string& decoded_string, float* likelihood, float* am_score, float* lm_score, float* confidence, float* expected_error_rate) override;

    protected:

        AgfNNet3OnlineModelConfig::Ptr config_;

        // Model
        StdConstFst *top_fst_ = nullptr;
        StdConstFst *dictation_fst_ = nullptr;
        std::unordered_map<int32, std::unique_ptr<StdConstFst>> grammar_fsts_;
        std::unordered_map<const StdFst*, std::string> grammar_fsts_name_map_;  // maps grammar_fst -> name; for debugging
        // INVARIANT: same size: grammar_fsts_, grammar_fsts_name_map_

        // Model objects
        ActiveGrammarFst* active_grammar_fst_ = nullptr;

        // Decoder objects
        SingleUtteranceNnet3DecoderTpl<fst::ActiveGrammarFst>* decoder_ = nullptr;  // reinstantiated per utterance
        CombineRuleNontermMapper<CompactLatticeArc>* rule_relabel_mapper_ = nullptr;

        bool IsUtteranceInProgress() const override { return DecoderReady(decoder_); }
        bool InvalidateActiveGrammarFst();
        void StartDecoding() override;
        void CleanupDecoder() override;
};

} // namespace dragonfly
