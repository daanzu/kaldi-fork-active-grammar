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
#include "lat/compose-lattice-pruned.h"
#include "lat/confidence.h"
#include "lat/lattice-functions.h"
#include "lat/sausages.h"
#include "lat/word-align-lattice-lexicon.h"
#include "lm/const-arpa-lm.h"
#include "rnnlm/rnnlm-lattice-rescoring.h"
#include "nnet3/nnet-utils.h"
#include "decoder/active-grammar-fst.h"

#include "utils.h"
#include "kaldi-utils.h"
#include "nlohmann_json.hpp"

#define DEFAULT_VERBOSITY 0

#define BEGIN_INTERFACE_CATCH_HANDLER \
    try {
#define END_INTERFACE_CATCH_HANDLER(expr) \
    } catch(const std::exception& e) { \
        KALDI_WARN << "Trying to survive fatal exception: " << e.what(); \
        return (expr); \
    }

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
    BaseFloat silence_weight = 1.0;  // default (1.0) means silence weighting disabled
    int32 frame_subsampling_factor = 3;
    std::string model_dir;
    std::string mfcc_config_filename;
    std::string ie_config_filename;
    std::string silence_phones_str = "1:2:3:4:5:6:7:8:9:10:11:12:13:14:15";  // FIXME: from lang/phones/silence.csl
    std::string model_filename;
    std::string word_syms_filename;
    std::string word_align_lexicon_filename;
    bool enable_ivector = true;
    bool enable_online_cmvn = false;
    std::string online_cmvn_config_filename;  // frequently file exists but is empty (except for comment)
    std::string orig_grammar_filename;
    bool enable_carpa = false;
    std::string carpa_filename;
    bool enable_rnnlm = false;
    std::string rnnlm_nnet_filename;
    std::string rnnlm_word_embed_filename;
    std::string ivector_extraction_config_json;  // extracted from ie_config_filename

    virtual bool Set(const std::string& name, const nlohmann::json& value) {
        if (name == "beam") { value.get_to(beam); return true; }
        if (name == "max_active") { value.get_to(max_active); return true; }
        if (name == "min_active") { value.get_to(min_active); return true; }
        if (name == "lattice_beam") { value.get_to(lattice_beam); return true; }
        if (name == "acoustic_scale") { value.get_to(acoustic_scale); return true; }
        if (name == "lm_weight") { value.get_to(lm_weight); return true; }
        if (name == "silence_weight") { value.get_to(silence_weight); return true; }
        if (name == "frame_subsampling_factor") { value.get_to(frame_subsampling_factor); return true; }
        if (name == "model_dir") { value.get_to(model_dir); return true; }
        if (name == "mfcc_config_filename") { value.get_to(mfcc_config_filename); return true; }
        if (name == "ie_config_filename") { value.get_to(ie_config_filename); return true; }
        if (name == "silence_phones_str") { value.get_to(silence_phones_str); return true; }
        if (name == "model_filename") { value.get_to(model_filename); return true; }
        if (name == "word_syms_filename") { value.get_to(word_syms_filename); return true; }
        if (name == "word_align_lexicon_filename") { value.get_to(word_align_lexicon_filename); return true; }
        if (name == "enable_ivector") { value.get_to(enable_ivector); return true; }
        if (name == "enable_online_cmvn") { value.get_to(enable_online_cmvn); return true; }
        if (name == "online_cmvn_config_filename") { value.get_to(online_cmvn_config_filename); return true; }
        if (name == "orig_grammar_filename") { value.get_to(orig_grammar_filename); return true; }
        if (name == "enable_carpa") { value.get_to(enable_carpa); return true; }
        if (name == "carpa_filename") { value.get_to(carpa_filename); return true; }
        if (name == "enable_rnnlm") { value.get_to(enable_rnnlm); return true; }
        if (name == "rnnlm_nnet_filename") { value.get_to(rnnlm_nnet_filename); return true; }
        if (name == "rnnlm_word_embed_filename") { value.get_to(rnnlm_word_embed_filename); return true; }
        if (name == "ivector_extraction_config_json") { ivector_extraction_config_json = value.dump(); return true; }
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
        ss << "\n    " << "enable_ivector: " << enable_ivector;
        ss << "\n    " << "enable_online_cmvn: " << enable_online_cmvn;
        ss << "\n    " << "online_cmvn_config_filename: " << online_cmvn_config_filename;
        ss << "\n    " << "orig_grammar_filename: " << orig_grammar_filename;
        ss << "\n    " << "enable_carpa: " << enable_carpa;
        ss << "\n    " << "carpa_filename: " << carpa_filename;
        ss << "\n    " << "enable_rnnlm: " << enable_rnnlm;
        ss << "\n    " << "rnnlm_nnet_filename: " << rnnlm_nnet_filename;
        ss << "\n    " << "rnnlm_word_embed_filename: " << rnnlm_word_embed_filename;
        ss << "\n    " << "ivector_extraction_config_json: " << ivector_extraction_config_json;
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

        bool SaveAdaptationState();  // Handles ivector-adaptation and online-cmvn
        void ResetAdaptationState();  // Handles ivector-adaptation and online-cmvn
        virtual bool GetWordAlignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths, bool include_eps);
        void SetLmPrimeText(const std::string& prime_text) { lm_prime_text_ = prime_text; };

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
        fst::SymbolTable *word_syms_ = nullptr;  // Word symbol table
        std::vector<std::vector<int32> > word_align_lexicon_;  // For each word, its word-id + word-id + a list of its phones

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
        WordAlignLatticeLexiconInfo* word_align_lexicon_info_ = nullptr;
        std::set<int32> word_align_lexicon_words_;  // contains word-ids that are in word_align_lexicon_info_

        // Ivector
        bool enable_ivector_ = false;
        OnlineIvectorExtractorAdaptationState* adaptation_state_ = nullptr;

        // Online-CMVN
        bool enable_online_cmvn_ = false;
        Matrix<double> global_cmvn_stats_;
        OnlineCmvnState* online_cmvn_state_ = nullptr;

        // CARPA
        bool enable_carpa_ = false;
        fst::MapFst<fst::StdArc, kaldi::LatticeArc, fst::StdToLatticeMapper<kaldi::BaseFloat> >* unscore_lm_fst_ = nullptr;
        ConstArpaLm carpa_;
        BaseFloat carpa_scale_ = 1.0;

        // RNNLM
        bool enable_rnnlm_ = false;
        nnet3::Nnet rnnlm_;
        CuMatrix<BaseFloat> word_embedding_mat_;
        fst::ScaleDeterministicOnDemandFst* lm_to_subtract_det_scale_ = nullptr;
        rnnlm::RnnlmComputeStateComputationOptions rnnlm_opts_;
        rnnlm::RnnlmComputeStateInfo* rnnlm_info_ = nullptr;
        int32 rnnlm_max_ngram_order_;
        ComposeLatticePrunedOptions rnnlm_compose_opts_;
        BaseFloat rnnlm_scale_ = 1.0;
        std::string lm_prime_text_;

        // Miscellaneous
        int32 tot_frames_ = 0, tot_frames_decoded_ = 0;
        bool decoder_finalized_ = false;
        CompactLattice decoded_clat_;
        CompactLattice best_path_clat_;

        StdConstFst* ReadFstFile(std::string filename);
        std::string WordIdsToString(const std::vector<int32> &wordIds);
        void RescoreConstArpaLm(CompactLattice& clat);
        void RescoreRnnlm(CompactLattice& clat, const std::string& prime_text = "");

        virtual void StartDecoding();
        virtual void CleanupDecoder();
};

} // namespace dragonfly
