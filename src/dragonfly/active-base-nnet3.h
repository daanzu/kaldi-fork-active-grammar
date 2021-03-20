// NNet3 Active Base

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
#include "nlohmann_json.hpp"

namespace dragonfly {

using namespace kaldi;
using namespace fst;


struct ActiveBaseNNet3OnlineModelConfig : public BaseNNet3OnlineModelConfig {
    using Ptr = std::shared_ptr<ActiveBaseNNet3OnlineModelConfig>;

    static constexpr auto Create = BaseNNet3OnlineModelConfig::Create<ActiveBaseNNet3OnlineModelConfig>;

    std::string dictation_fst_filename;
    int32 max_num_rules = 10000;
    int32 max_num_exported_rules = 1000;

    bool Set(const std::string& name, const nlohmann::json& value) override {
        if (BaseNNet3OnlineModelConfig::Set(name, value)) { return true; }
        if (name == "dictation_fst_filename") { value.get_to(dictation_fst_filename); return true; }
        if (name == "max_num_rules") { value.get_to(max_num_rules); return true; }
        if (name == "max_num_exported_rules") { value.get_to(max_num_exported_rules); return true; }
        return false;
    }

    std::string ToString() override {
        stringstream ss;
        ss << BaseNNet3OnlineModelConfig::ToString() << '\n';
        ss << "ActiveBaseNNet3OnlineModelConfig...";
        ss << "\n    " << "dictation_fst_filename: " << dictation_fst_filename;
        ss << "\n    " << "max_num_rules: " << max_num_rules;
        ss << "\n    " << "max_num_exported_rules: " << max_num_exported_rules;
        return ss.str();
    }
};

class ActiveBaseNNet3OnlineModelWrapper : public BaseNNet3OnlineModelWrapper {
    public:

        ActiveBaseNNet3OnlineModelWrapper(ActiveBaseNNet3OnlineModelConfig::Ptr config, int32 verbosity = DEFAULT_VERBOSITY);
        ~ActiveBaseNNet3OnlineModelWrapper() override;

        void SetActiveGrammars(std::set<int32>& grammars_activity) { if (grammars_activity_ != grammars_activity) grammars_activity_.swap(grammars_activity); };

        bool SetMimicGrammarFst(int32 grammar_fst_index, StdFst* grammar_fst);
        bool Mimic(const std::string& input, std::string* output_p) { return MimicInternal(input, output_p, -1); }
        bool MimicGrammar(const std::string& input, std::string* output_p, int32 grammar_fst_index) {
            if (grammar_fst_index < 0) KALDI_ERR << "Invalid grammar_fst_index";
            return MimicInternal(input, output_p, grammar_fst_index);
        }

    protected:

        ActiveBaseNNet3OnlineModelConfig::Ptr config_;

        std::set<int32> grammars_activity_;  // Grammar rule numbers (local indices) that are active for current/upcoming utterance.

        std::unordered_map<int32, std::shared_ptr<StdConstFst>> mimic_fsts_;
        std::shared_ptr<StdConstFst> mimic_dictation_fst_;

        bool MimicInternal(const std::string& input, std::string* output_p, int32 grammar_fst_index);
};

} // namespace dragonfly
