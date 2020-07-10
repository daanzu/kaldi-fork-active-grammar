// NNet3 Plain

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

struct PlainNNet3OnlineModelConfig : public BaseNNet3OnlineModelConfig {

    std::string decode_fst_filename;

    bool Set(const std::string& name, const nlohmann::json& value) {
        if (BaseNNet3OnlineModelConfig::Set(name, value)) { return true; }
        if (name == "decode_fst_filename") { decode_fst_filename = value.get<std::string>(); return true; }
        return false;
    }

    std::string ToString() {
        stringstream ss;
        ss << BaseNNet3OnlineModelConfig::ToString() << '\n';
        ss << "PlainNNet3OnlineModelConfig...";
        ss << "\n    " << "decode_fst_filename: " << decode_fst_filename;
        return ss.str();
    }
};

class PlainNNet3OnlineModelWrapper : public BaseNNet3OnlineModelWrapper {
    public:

        PlainNNet3OnlineModelWrapper(const std::string& model_dir, const std::string& config_str = "", int32 verbosity = DEFAULT_VERBOSITY);
        ~PlainNNet3OnlineModelWrapper();

        bool Decode(BaseFloat samp_freq, const Vector<BaseFloat>& frames, bool finalize, bool save_adaptation_state = true);
        virtual void GetDecodedString(std::string& decoded_string, float* likelihood, float* am_score, float* lm_score, float* confidence, float* expected_error_rate);
        virtual bool GetWordAlignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths, bool include_eps);

    protected:

        PlainNNet3OnlineModelConfig config_;

        // Model objects
        StdConstFst* decode_fst_ = nullptr;

        // Decoder objects
        SingleUtteranceNnet3Decoder* decoder_ = nullptr;  // reinstantiated per utterance

        void StartDecoding() override;
        void CleanupDecoder() override;
};

}  // namespace dragonfly
