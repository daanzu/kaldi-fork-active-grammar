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

    std::string hcl_fst_filename;
    std::string disambig_tids_filename;
    std::string dictation_fst_filename;

    bool Set(const std::string& name, const nlohmann::json& value) override {
        if (BaseNNet3OnlineModelConfig::Set(name, value)) { return true; }
        if (name == "hcl_fst_filename") { hcl_fst_filename = value.get<std::string>(); return true; }
        if (name == "disambig_tids_filename") { disambig_tids_filename = value.get<std::string>(); return true; }
        if (name == "dictation_fst_filename") { dictation_fst_filename = value.get<std::string>(); return true; }
        return false;
    }

    std::string ToString() override {
        stringstream ss;
        ss << BaseNNet3OnlineModelConfig::ToString() << '\n';
        ss << "LafNNet3OnlineModelConfig...";
        ss << "\n    " << "hcl_fst_filename: " << hcl_fst_filename;
        ss << "\n    " << "disambig_tids_filename: " << disambig_tids_filename;
        ss << "\n    " << "dictation_fst_filename: " << dictation_fst_filename;
        return ss.str();
    }
};

class LafNNet3OnlineModelWrapper : public BaseNNet3OnlineModelWrapper {
    public:

        LafNNet3OnlineModelWrapper(LafNNet3OnlineModelConfig::Ptr config, int32 verbosity = DEFAULT_VERBOSITY);
        ~LafNNet3OnlineModelWrapper() override;

        virtual int32 AddGrammarFst(std::string& grammar_fst_filename);
        virtual bool ReloadGrammarFst(int32 grammar_fst_index, std::string& grammar_fst_filename);
        virtual bool RemoveGrammarFst(int32 grammar_fst_index);
        void SetActiveGrammars(const std::vector<bool>& grammars_activity) { grammars_activity_ = grammars_activity; };

        bool Decode(BaseFloat samp_freq, const Vector<BaseFloat>& frames, bool finalize, const std::vector<bool>& grammars_activity, bool save_adaptation_state = true);
        bool Decode(BaseFloat samp_freq, const Vector<BaseFloat>& frames, bool finalize, bool save_adaptation_state = true) override;
        void GetDecodedString(std::string& decoded_string, float* likelihood, float* am_score, float* lm_score, float* confidence, float* expected_error_rate) override;

    protected:

        LafNNet3OnlineModelConfig::Ptr config_;

        // Model
        StdFst *hcl_fst_ = nullptr;
        std::vector<int32> disambig_tids_;
        StdConstFst *dictation_fst_ = nullptr;
        std::vector<StdFst*> grammar_fsts_;
        std::map<StdFst*, std::string> grammar_fsts_filename_map_;  // maps grammar_fst -> name; for debugging
        // INVARIANT: same size: grammar_fsts_, grammar_fsts_filename_map_
        std::vector<bool> grammars_activity_;  // bitfield of whether each grammar is active for current/upcoming utterance

        // Model objects
        // ActiveGrammarFst* active_grammar_fst_ = nullptr;

        // Decoder objects
        SingleUtteranceNnet3DecoderTpl<fst::StdFst>* decoder_ = nullptr;  // reinstantiated per utterance
        CombineRuleNontermMapper<CompactLatticeArc>* rule_relabel_mapper_ = nullptr;

        void StartDecoding() override;
        void CleanupDecoder() override;
};

} // namespace dragonfly
