// NNet3 Base

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

#include "utils.h"
#include "kaldi-utils.h"
#include "nlohmann_json.hpp"

#define DEFAULT_VERBOSITY 0

namespace dragonfly {

using namespace kaldi;
using namespace fst;

struct BaseNNet3OnlineModelConfig {
    using Ptr = std::shared_ptr<BaseNNet3OnlineModelConfig>;

    BaseFloat beam = 14.0;  // normally 7.0
    int32 max_active = 14000;  // normally 7000
    int32 min_active = 200;
    BaseFloat lattice_beam = 8.0;
    BaseFloat acoustic_scale = 1.0;
    BaseFloat lm_weight = 7.0;  // 10.0 would be "neutral", with no scaling
    BaseFloat silence_weight = 1.0;  // default means silence weighting disabled
    int32 frame_subsampling_factor = 3;
    std::string model_dir;
    std::string mfcc_config_filename;
    std::string ie_config_filename;
    std::string silence_phones_str = "1:2:3:4:5:6:7:8:9:10:11:12:13:14:15";  // FIXME: from lang/phones/silence.csl
    std::string model_filename;
    std::string word_syms_filename;
    std::string word_align_lexicon_filename;

    virtual bool Set(const std::string& name, const nlohmann::json& value) {
        if (name == "beam") { beam = value.get<BaseFloat>(); return true; }
        if (name == "max_active") { max_active = value.get<int32>(); return true; }
        if (name == "min_active") { min_active = value.get<int32>(); return true; }
        if (name == "lattice_beam") { lattice_beam = value.get<BaseFloat>(); return true; }
        if (name == "acoustic_scale") { acoustic_scale = value.get<BaseFloat>(); return true; }
        if (name == "lm_weight") { lm_weight = value.get<BaseFloat>(); return true; }
        if (name == "silence_weight") { silence_weight = value.get<BaseFloat>(); return true; }
        if (name == "frame_subsampling_factor") { frame_subsampling_factor = value.get<int32>(); return true; }
        if (name == "model_dir") { model_dir = value.get<std::string>(); return true; }
        if (name == "mfcc_config_filename") { mfcc_config_filename = value.get<std::string>(); return true; }
        if (name == "ie_config_filename") { ie_config_filename = value.get<std::string>(); return true; }
        if (name == "silence_phones_str") { silence_phones_str = value.get<std::string>(); return true; }
        if (name == "model_filename") { model_filename = value.get<std::string>(); return true; }
        if (name == "word_syms_filename") { word_syms_filename = value.get<std::string>(); return true; }
        if (name == "word_align_lexicon_filename") { word_align_lexicon_filename = value.get<std::string>(); return true; }
        return false;
    }

    virtual std::string ToString() {
        stringstream ss;
        ss << "BaseNNet3OnlineModelConfig...";
        ss << "\n    " << "beam: " << beam;
        ss << "\n    " << "max_active: " << max_active;
        ss << "\n    " << "min_active: " << min_active;
        ss << "\n    " << "lattice_beam: " << lattice_beam;
        ss << "\n    " << "acoustic_scale: " << acoustic_scale;
        ss << "\n    " << "lm_weight: " << lm_weight;
        ss << "\n    " << "silence_weight: " << silence_weight;
        ss << "\n    " << "frame_subsampling_factor: " << frame_subsampling_factor;
        ss << "\n    " << "model_dir: " << model_dir;
        ss << "\n    " << "mfcc_config_filename: " << mfcc_config_filename;
        ss << "\n    " << "ie_config_filename: " << ie_config_filename;
        ss << "\n    " << "silence_phones_str: " << silence_phones_str;
        ss << "\n    " << "model_filename: " << model_filename;
        ss << "\n    " << "word_syms_filename: " << word_syms_filename;
        ss << "\n    " << "word_align_lexicon_filename: " << word_align_lexicon_filename;
        return ss.str();
    }

    template <class Config>
    static std::shared_ptr<Config> Create(const std::string& model_dir_str, const std::string& config_str = "") {
        auto config = std::make_shared<Config>();
        if (model_dir_str.empty())
            KALDI_ERR << "Empty model_dir";
        config->model_dir = model_dir_str;
        if (!config_str.empty()) {
            auto config_json = nlohmann::json::parse(config_str);
            if (!config_json.is_object())
                KALDI_ERR << "config_str must be a valid JSON object";
            for (const auto& it : config_json.items()) {
                if (!config->Set(it.key(), it.value()))
                    KALDI_WARN << "Bad config key: " << it.key() << " = " << it.value();
            }
        }
        return config;
    }
};

class BaseNNet3OnlineModelWrapper {
    public:

        BaseNNet3OnlineModelWrapper(BaseNNet3OnlineModelConfig::Ptr config, int32 verbosity = DEFAULT_VERBOSITY);
        virtual ~BaseNNet3OnlineModelWrapper();

        bool LoadLexicon(std::string& word_syms_filename, std::string& word_align_lexicon_filename);

        bool SaveAdaptationState();
        void ResetAdaptationState();
        virtual bool GetWordAlignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths, bool include_eps);

        virtual bool Decode(BaseFloat samp_freq, const Vector<BaseFloat>& frames, bool finalize, bool save_adaptation_state = true) = 0;
        virtual void GetDecodedString(std::string& decoded_string, float* likelihood, float* am_score, float* lm_score, float* confidence, float* expected_error_rate) = 0;

    protected:

        // Templated decode methods
        template <typename Decoder>
        bool Decode(Decoder* decoder, BaseFloat samp_freq, const Vector<BaseFloat>& frames, bool finalize, bool save_adaptation_state = true);
        template <typename Decoder>
        bool DecoderReady(Decoder* decoder) const { return (decoder && !decoder_finalized_); };

        BaseNNet3OnlineModelConfig::Ptr config_;

        // Model
        fst::SymbolTable *word_syms_ = nullptr;
        std::vector<std::vector<int32> > word_align_lexicon_;  // for each word, its word-id + word-id + a list of its phones

        // Model objects
        OnlineNnet2FeaturePipelineConfig feature_config_;
        nnet3::NnetSimpleLoopedComputationOptions decodable_config_;
        LatticeFasterDecoderConfig decoder_config_;
        OnlineEndpointConfig endpoint_config_;
        TransitionModel trans_model_;
        nnet3::AmNnetSimple am_nnet_;
        OnlineNnet2FeaturePipelineInfo* feature_info_ = nullptr;  // TODO: doesn't really need to be dynamically allocated (pointer)
        nnet3::DecodableNnetSimpleLoopedInfo* decodable_info_ = nullptr;  // contains precomputed stuff that is used by all decodable objects

        // Decoder objects
        OnlineNnet2FeaturePipeline* feature_pipeline_ = nullptr;  // reinstantiated per utterance
        OnlineSilenceWeighting* silence_weighting_ = nullptr;  // reinstantiated per utterance
        OnlineIvectorExtractorAdaptationState* adaptation_state_ = nullptr;
        WordAlignLatticeLexiconInfo* word_align_lexicon_info_ = nullptr;
        std::set<int32> word_align_lexicon_words_;  // contains word-ids that are in word_align_lexicon_info_

        int32 tot_frames_ = 0, tot_frames_decoded_ = 0;
        bool decoder_finalized_ = false;
        CompactLattice decoded_clat_;
        CompactLattice best_path_clat_;

        StdConstFst* ReadFstFile(std::string filename);
        std::string WordIdsToString(const std::vector<int32> &wordIds);

        virtual void StartDecoding();
        virtual void CleanupDecoder();
};

} // namespace dragonfly
