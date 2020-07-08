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
    std::string model_filename;
    int32 nonterm_phones_offset = -1;  // offset from start of phones that start of nonterms are
    int32 rules_phones_offset = -1;  // offset from start of phones that the dictation nonterms are
    int32 dictation_phones_offset = -1;  // offset from start of phones that the kaldi_rules nonterms are
    std::string silence_phones_str = "1:2:3:4:5:6:7:8:9:10:11:12:13:14:15";  // FIXME: from lang/phones/silence.csl
    std::string word_syms_filename;
    std::string word_align_lexicon_filename;
    std::string top_fst_filename;
    std::string dictation_fst_filename;

    bool Set(const std::string& name, const nlohmann::json& value) {
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
        if (name == "model_filename") { model_filename = value.get<std::string>(); return true; }
        if (name == "nonterm_phones_offset") { nonterm_phones_offset = value.get<int32>(); return true; }
        if (name == "rules_phones_offset") { rules_phones_offset = value.get<int32>(); return true; }
        if (name == "dictation_phones_offset") { dictation_phones_offset = value.get<int32>(); return true; }
        if (name == "silence_phones_str") { silence_phones_str = value.get<std::string>(); return true; }
        if (name == "word_syms_filename") { word_syms_filename = value.get<std::string>(); return true; }
        if (name == "word_align_lexicon_filename") { word_align_lexicon_filename = value.get<std::string>(); return true; }
        if (name == "top_fst_filename") { top_fst_filename = value.get<std::string>(); return true; }
        if (name == "dictation_fst_filename") { dictation_fst_filename = value.get<std::string>(); return true; }
        return false;
    }

    std::string ToString() {
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
        ss << "\n    " << "model_filename: " << model_filename;
        ss << "\n    " << "nonterm_phones_offset: " << nonterm_phones_offset;
        ss << "\n    " << "rules_phones_offset: " << rules_phones_offset;
        ss << "\n    " << "dictation_phones_offset: " << dictation_phones_offset;
        ss << "\n    " << "silence_phones_str: " << silence_phones_str;
        ss << "\n    " << "word_syms_filename: " << word_syms_filename;
        ss << "\n    " << "word_align_lexicon_filename: " << word_align_lexicon_filename;
        ss << "\n    " << "top_fst_filename: " << top_fst_filename;
        ss << "\n    " << "dictation_fst_filename: " << dictation_fst_filename;
        return ss.str();
    }
};

class BaseNNet3OnlineModelWrapper {
    public:

        BaseNNet3OnlineModelWrapper(const std::string& model_dir, const std::string& config_str = "", int32 verbosity = DEFAULT_VERBOSITY);
        ~BaseNNet3OnlineModelWrapper();

        bool LoadLexicon(std::string& word_syms_filename, std::string& word_align_lexicon_filename);
        int32 AddGrammarFst(std::string& grammar_fst_filename);
        bool ReloadGrammarFst(int32 grammar_fst_index, std::string& grammar_fst_filename);
        bool RemoveGrammarFst(int32 grammar_fst_index);
        bool SaveAdaptationState();
        void ResetAdaptationState();
        bool Decode(BaseFloat samp_freq, const Vector<BaseFloat>& frames, bool finalize, std::vector<bool>& grammars_activity, bool save_adaptation_state = true);

        void GetDecodedString(std::string& decoded_string, float* likelihood, float* am_score, float* lm_score, float* confidence, float* expected_error_rate);
        bool GetWordAlignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths, bool include_eps);

    protected:

        BaseNNet3OnlineModelConfig config_;

        // Model
        fst::SymbolTable *word_syms_ = nullptr;
        std::vector<std::vector<int32> > word_align_lexicon_;  // for each word, its word-id + word-id + a list of its phones
        StdConstFst *top_fst_ = nullptr;
        StdConstFst *dictation_fst_ = nullptr;
        std::vector<StdConstFst*> grammar_fsts_;
        std::map<StdConstFst*, std::string> grammar_fsts_filename_map_;  // maps grammar_fst -> name; for debugging
        // INVARIANT: same size: grammar_fsts_, grammar_fsts_filename_map_

        // Model objects
        OnlineNnet2FeaturePipelineConfig feature_config_;
        nnet3::NnetSimpleLoopedComputationOptions decodable_config_;
        LatticeFasterDecoderConfig decoder_config_;
        OnlineEndpointConfig endpoint_config_;
        TransitionModel trans_model_;
        nnet3::AmNnetSimple am_nnet_;
        OnlineNnet2FeaturePipelineInfo* feature_info_ = nullptr;  // TODO: doesn't really need to be dynamically allocated (pointer)
        nnet3::DecodableNnetSimpleLoopedInfo* decodable_info_ = nullptr;  // contains precomputed stuff that is used by all decodable objects
        ActiveGrammarFst* active_grammar_fst_ = nullptr;

        // Decoder objects
        OnlineNnet2FeaturePipeline* feature_pipeline_ = nullptr;  // reinstantiated per utterance
        OnlineSilenceWeighting* silence_weighting_ = nullptr;  // reinstantiated per utterance
        OnlineIvectorExtractorAdaptationState* adaptation_state_ = nullptr;
        SingleUtteranceNnet3DecoderTpl<fst::ActiveGrammarFst>* decoder_ = nullptr;  // reinstantiated per utterance
        WordAlignLatticeLexiconInfo* word_align_lexicon_info_ = nullptr;
        std::set<int32> word_align_lexicon_words_;  // contains word-ids that are in word_align_lexicon_info_
        CombineRuleNontermMapper<CompactLatticeArc>* rule_relabel_mapper_ = nullptr;

        int32 tot_frames = 0, tot_frames_decoded = 0;
        bool decoder_finalized_ = false;
        CompactLattice decoded_clat_;
        CompactLattice best_path_clat_;

        StdConstFst* ReadFstFile(std::string filename);
        void StartDecoding(std::vector<bool> grammars_activity);
        void CleanupDecoder();
        std::string WordIdsToString(const std::vector<int32> &wordIds);
};

}  // namespace dragonfly
