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
#include "lm/const-arpa-lm.h"
#include "rnnlm/rnnlm-lattice-rescoring.h"
#include "nnet3/nnet-utils.h"
#include "decoder/active-grammar-fst.h"

#include "base-nnet3.h"
#include "utils.h"
#include "kaldi-utils.h"
#include "nlohmann_json.hpp"

namespace dragonfly {

using namespace kaldi;
using namespace fst;

BaseNNet3OnlineModelWrapper::BaseNNet3OnlineModelWrapper(BaseNNet3OnlineModelConfig::Ptr config_arg, int32 verbosity) : config_(std::move(config_arg)) {
    SetVerboseLevel(verbosity);
    if (verbosity >= 0) {
        KALDI_LOG << "Verbosity: " << verbosity;
    } else if (verbosity == -1) {
        SetLogHandler([](const LogMessageEnvelope& envelope, const char* message) {
            if (envelope.severity <= LogMessageEnvelope::kWarning) {
                std::cerr << "[KALDI severity=" << envelope.severity << "] " << message << "\n";
            }
        });
    } else {
        // Silence kaldi output as well (even warnings and errors!)
        SetLogHandler([](const LogMessageEnvelope& envelope, const char* message) {});
    }

    KALDI_LOG << config_->ToString();

    if (true && verbosity >= 1) {
        ExecutionTimer timer("testing output latency");
        std::cerr << "[testing output latency][testing output latency][testing output latency]" << endl;
    }

    ExecutionTimer timer("Initialization/loading");

    enable_ivector_ = config_->enable_ivector;
    if (!enable_ivector_) KALDI_ERR << "Disabling ivector not tested!";

    if (!config_->ivector_extraction_config_json.empty()) {
        // Load ivector-extractor from json passed in config, directly into constructed object.
        feature_info_ = new OnlineNnet2FeaturePipelineInfo();  // starts with defaults

        // From BaseNNet3OnlineModelConfig.mfcc_config_filename, aka OnlineNnet2FeaturePipelineConfig.mfcc_config
        ReadConfigFromFile(config_->mfcc_config_filename, &feature_info_->mfcc_opts);

        feature_info_->use_ivectors = config_->enable_ivector;
        // From BaseNNet3OnlineModelConfig.ie_config_filename, aka OnlineNnet2FeaturePipelineConfig.ivector_extraction_config
        auto ivector_extraction_config = nlohmann::json::parse(config_->ivector_extraction_config_json).get<OnlineIvectorExtractionConfig>();
        feature_info_->ivector_extractor_info.Init(ivector_extraction_config);

        enable_online_cmvn_ = config_->enable_online_cmvn;
        if (enable_online_cmvn_) {
            feature_info_->use_cmvn = true;
            if (!config_->online_cmvn_config_filename.empty())
                // From BaseNNet3OnlineModelConfig.online_cmvn_config_filename, aka OnlineNnet2FeaturePipelineConfig.cmvn_config.
                ReadConfigFromFile(config_->online_cmvn_config_filename, &feature_info_->cmvn_opts);
            // If config filename is empty, assume the original file itself was empty as well, and thus left options at defaults.
            // We could set feature_info_->cmvn_opts members directly ourselves, after starting with the defaults.
            if (ivector_extraction_config.global_cmvn_stats_rxfilename.empty()) KALDI_ERR << "Must give global_cmvn_stats_rxfilename";
            feature_info_->global_cmvn_stats_rxfilename = ivector_extraction_config.global_cmvn_stats_rxfilename;
            ReadKaldiObject(feature_info_->global_cmvn_stats_rxfilename, &global_cmvn_stats_);
            // global_cmvn_stats_ = feature_info_->ivector_extractor_info.global_cmvn_stats;  // Just copy from ivector, since we have it
            if (!ivector_extraction_config.online_cmvn_iextractor) KALDI_ERR << "enable_online_cmvn_ is true, but ivector_extraction_config.online_cmvn_iextractor is false";
            if (!feature_info_->ivector_extractor_info.online_cmvn_iextractor) KALDI_ERR << "enable_online_cmvn_ is true, but feature_info_->ivector_extractor_info.online_cmvn_iextractor is false";
            feature_info_->ivector_extractor_info.online_cmvn_iextractor = true;
        } else {
            if (feature_info_->ivector_extractor_info.online_cmvn_iextractor) KALDI_ERR << "enable_online_cmvn_ is false, but feature_info_->ivector_extractor_info.online_cmvn_iextractor is true";
        }

        feature_info_->silence_weighting_config.silence_weight = config_->silence_weight;
        feature_info_->silence_weighting_config.silence_phones_str = config_->silence_phones_str;

    } else {
        // Deprecated rewritten-file configuration.
        feature_config_.mfcc_config = config_->mfcc_config_filename;
        feature_config_.ivector_extraction_config = config_->ie_config_filename;
        feature_config_.silence_weighting_config.silence_weight = config_->silence_weight;
        feature_config_.silence_weighting_config.silence_phones_str = config_->silence_phones_str;
        feature_info_ = new OnlineNnet2FeaturePipelineInfo(feature_config_);
        if (config_->enable_online_cmvn) KALDI_WARN << "online-cmvn not supported with this configuration";
    }

    {
        bool binary;
        Input ki(config_->model_filename, &binary);
        trans_model_.Read(ki.Stream(), binary);
        am_nnet_.Read(ki.Stream(), binary);
        SetBatchnormTestMode(true, &(am_nnet_.GetNnet()));
        SetDropoutTestMode(true, &(am_nnet_.GetNnet()));
        nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet_.GetNnet()));
    }

    decodable_config_.acoustic_scale = config_->acoustic_scale;
    decodable_config_.frame_subsampling_factor = config_->frame_subsampling_factor;
    decodable_info_ = new nnet3::DecodableNnetSimpleLoopedInfo(decodable_config_, &am_nnet_);
    decoder_config_.max_active = config_->max_active;
    decoder_config_.min_active = config_->min_active;
    decoder_config_.beam = config_->beam;
    decoder_config_.lattice_beam = config_->lattice_beam;
    ResetAdaptationState();

    LoadLexicon(config_->word_syms_filename, config_->word_align_lexicon_filename);

    enable_carpa_ = config_->enable_carpa;
    if (enable_carpa_) {
        ExecutionTimer timer("loading carpa");
        if (config_->orig_grammar_filename.empty()) KALDI_ERR << "orig_grammar_filename not set";
        if (config_->carpa_filename.empty()) KALDI_ERR << "carpa_filename not set";
        ReadKaldiObject(config_->model_dir + "/" + config_->carpa_filename, &carpa_);

        VectorFst<StdArc> *lm_to_subtract_fst = ReadAndPrepareLmFst(config_->model_dir + "/" + config_->orig_grammar_filename);  // FIXME: manage this and delete it, and share
        fst::CacheOptions cache_opts(true, 50000000);  // Faster than 50000?
        fst::MapFstOptions mapfst_opts(cache_opts);
        fst::StdToLatticeMapper<kaldi::BaseFloat> mapper;
        unscore_lm_fst_ = new fst::MapFst<fst::StdArc, kaldi::LatticeArc, fst::StdToLatticeMapper<kaldi::BaseFloat> >(*lm_to_subtract_fst, mapper, mapfst_opts);
    } else if (!config_->carpa_filename.empty())
        KALDI_ERR << "enable_carpa_ is false, but some carpa options are set";

    enable_rnnlm_ = config_->enable_rnnlm;
    if (enable_rnnlm_) {
        ExecutionTimer timer("loading rnnlm");
        if (config_->orig_grammar_filename.empty()) KALDI_ERR << "orig_grammar_filename not set";
        if (config_->rnnlm_nnet_filename.empty()) KALDI_ERR << "rnnlm_nnet_filename not set";
        if (config_->rnnlm_word_embed_filename.empty()) KALDI_ERR << "rnnlm_word_embed_filename not set";
        
        VectorFst<StdArc> *lm_to_subtract_fst = ReadAndPrepareLmFst(config_->model_dir + "/" + config_->orig_grammar_filename);  // FIXME: manage this and delete it, and share
        BackoffDeterministicOnDemandFst<StdArc> *lm_to_subtract_det_backoff = new BackoffDeterministicOnDemandFst<StdArc>(*lm_to_subtract_fst);
        lm_to_subtract_det_scale_ = new ScaleDeterministicOnDemandFst(-rnnlm_scale_, lm_to_subtract_det_backoff);

        ReadKaldiObject((config_->model_dir + "/" + config_->rnnlm_nnet_filename), &rnnlm_);
        KALDI_ASSERT(IsSimpleNnet(rnnlm_));
        ReadKaldiObject((config_->model_dir + "/" + config_->rnnlm_word_embed_filename), &word_embedding_mat_);

        rnnlm_opts_.bos_index = word_syms_->Find("<s>");
        rnnlm_opts_.eos_index = word_syms_->Find("</s>");
        rnnlm_info_ = new rnnlm::RnnlmComputeStateInfo(rnnlm_opts_, rnnlm_, word_embedding_mat_);
        rnnlm_max_ngram_order_ = 4;
    } else if (!config_->rnnlm_nnet_filename.empty() || !config_->rnnlm_word_embed_filename.empty())
        KALDI_ERR << "enable_rnnlm_ is false, but some rnnlm options are set";
    
    if (enable_carpa_ && enable_rnnlm_)
        KALDI_WARN << "are you sure you want to enable both CARPA and RNNLM rescoring?";
}

BaseNNet3OnlineModelWrapper::~BaseNNet3OnlineModelWrapper() {
    CleanupDecoder();
    delete word_syms_;
    delete feature_info_;
    delete decodable_info_;
    delete adaptation_state_;
    delete word_align_lexicon_info_;
}

bool BaseNNet3OnlineModelWrapper::LoadLexicon(std::string& word_syms_filename, std::string& word_align_lexicon_filename) {
    // FIXME: make more robust to errors

    if (word_syms_filename != "") {
        if (!(word_syms_ = fst::SymbolTable::ReadText(word_syms_filename))) {
            KALDI_ERR << "Could not read symbol table from file " << word_syms_filename;
            return false;
        }
    }

    if (word_align_lexicon_filename != "") {
        bool binary_in;
        Input ki(word_align_lexicon_filename, &binary_in);
        KALDI_ASSERT(!binary_in && "Not expecting binary file for lexicon");
        if (!ReadLexiconForWordAlign(ki.Stream(), &word_align_lexicon_)) {
            KALDI_ERR << "Error reading word alignment lexicon from file " << word_align_lexicon_filename;
            return false;
        }
        if (word_align_lexicon_info_)
            delete word_align_lexicon_info_;
        word_align_lexicon_info_ = new WordAlignLatticeLexiconInfo(word_align_lexicon_);

        word_align_lexicon_words_.clear();
        for (auto entry : word_align_lexicon_)
            word_align_lexicon_words_.insert(entry.at(0));
    }

    return true;
}

StdConstFst* BaseNNet3OnlineModelWrapper::ReadFstFile(std::string filename) {
    if (filename.compare(filename.length() - 4, 4, ".txt") == 0) {
        // TODO?: fstdeterminize | fstminimize | fstrmepsilon | fstarcsort --sort_type=ilabel
        KALDI_WARN << "cannot read text fst file " << filename;
        return nullptr;
    } else {
        auto fst = dynamic_cast<StdConstFst*>(ReadFstKaldiGeneric(filename));
        if (!fst) KALDI_ERR << "could not load as StdConstFst";
        return fst;
    }
}

std::string BaseNNet3OnlineModelWrapper::WordIdsToString(const std::vector<int32> &wordIds) {
    stringstream text;
    for (size_t i = 0; i < wordIds.size(); i++) {
        std::string s = word_syms_->Find(wordIds[i]);
        if (s == "") {
            KALDI_WARN << "Word-id " << wordIds[i] << " not in symbol table";
            s = "MISSING_WORD";
        }
        if (i != 0) text << " ";
        text << word_syms_->Find(wordIds[i]);
    }
    return text.str();
}

void BaseNNet3OnlineModelWrapper::StartDecoding() {
    // Cleanup
    CleanupDecoder();
    decoder_finalized_ = false;
    decoded_clat_.DeleteStates();
    best_path_clat_.DeleteStates();

    // Setup
    feature_pipeline_ = new OnlineNnet2FeaturePipeline(*feature_info_);
    if (enable_ivector_) feature_pipeline_->SetAdaptationState(*adaptation_state_);
    if (enable_online_cmvn_) feature_pipeline_->SetCmvnState(*online_cmvn_state_);
    silence_weighting_ = new OnlineSilenceWeighting(
        trans_model_, feature_info_->silence_weighting_config,
        decodable_config_.frame_subsampling_factor);
    // Child class should afterwards setup decoder
}

void BaseNNet3OnlineModelWrapper::CleanupDecoder() {
    delete silence_weighting_;
    silence_weighting_ = nullptr;
    delete feature_pipeline_;
    feature_pipeline_ = nullptr;
}

bool BaseNNet3OnlineModelWrapper::SaveAdaptationState() {
    if (feature_pipeline_) {
        if (enable_ivector_) feature_pipeline_->GetAdaptationState(adaptation_state_);
        if (enable_online_cmvn_) feature_pipeline_->GetCmvnState(online_cmvn_state_);
        KALDI_LOG << "Saved adaptation state.";
        return true;
    }
    return false;
}

void BaseNNet3OnlineModelWrapper::ResetAdaptationState() {
    delete adaptation_state_;
    adaptation_state_ = nullptr;
    if (enable_ivector_) adaptation_state_ = new OnlineIvectorExtractorAdaptationState(feature_info_->ivector_extractor_info);
    delete online_cmvn_state_;
    online_cmvn_state_ = nullptr;
    if (enable_online_cmvn_) online_cmvn_state_ = new OnlineCmvnState(global_cmvn_stats_);
}

bool BaseNNet3OnlineModelWrapper::GetWordAlignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths, bool include_eps) {
    if (!word_align_lexicon_.size() || !word_align_lexicon_info_) KALDI_ERR << "No word alignment lexicon loaded";
    if (best_path_clat_.NumStates() == 0) KALDI_ERR << "No best path lattice";

    // if (!best_path_has_valid_word_align) {
    //     KALDI_ERR << "There was a word not in word alignment lexicon";
    // }
    // if (!word_align_lexicon_words_.count(words[i])) {
    //     KALDI_LOG << "Word " << s << " (id #" << words[i] << ") not in word alignment lexicon";
    // }

    CompactLattice aligned_clat;
    WordAlignLatticeLexiconOpts opts;
    bool ok = WordAlignLatticeLexicon(best_path_clat_, trans_model_, *word_align_lexicon_info_, opts, &aligned_clat);

    if (!ok) {
        KALDI_WARN << "Lattice did not align correctly";
        return false;
    }

    if (aligned_clat.Start() == fst::kNoStateId) {
        KALDI_WARN << "Lattice was empty";
        return false;
    }

    TopSortCompactLatticeIfNeeded(&aligned_clat);

    // lattice-1best
    CompactLattice best_path_aligned;
    CompactLatticeShortestPath(aligned_clat, &best_path_aligned);

    // nbest-to-ctm
    std::vector<int32> word_idxs, times_raw, lengths_raw;
    ok = CompactLatticeToWordAlignment(best_path_aligned, &word_idxs, &times_raw, &lengths_raw);
    if (!ok) {
        KALDI_WARN << "CompactLatticeToWordAlignment failed.";
        return false;
    }

    // lexicon lookup
    words.clear();
    for (size_t i = 0; i < word_idxs.size(); i++) {
        std::string s = word_syms_->Find(word_idxs[i]);  // Must be found, or CompactLatticeToWordAlignment would have crashed
        // KALDI_LOG << "align: " << s << " - " << times_raw[i] << " - " << lengths_raw[i];
        if (include_eps || (word_idxs[i] != 0)) {
            words.push_back(s);
            times.push_back(times_raw[i]);
            lengths.push_back(lengths_raw[i]);
        }
    }
    return true;
}

void BaseNNet3OnlineModelWrapper::RescoreConstArpaLm(CompactLattice& clat) {
    ExecutionTimer timer("carpa rescoring");
    // See lattice-lmrescore.cc
    Lattice lat1;
    ConvertLattice(clat, &lat1);
    fst::ScaleLattice(fst::GraphLatticeScale(-1.0), &lat1);
    fst::ArcSort(&lat1, fst::OLabelCompare<kaldi::LatticeArc>());
    kaldi::Lattice composed_lat;
    fst::Compose(lat1, *unscore_lm_fst_, &composed_lat);
    fst::Invert(&composed_lat);
    kaldi::CompactLattice determinized_lat;
    DeterminizeLattice(composed_lat, &determinized_lat);
    fst::ScaleLattice(fst::GraphLatticeScale(-1.0), &determinized_lat);
    if (determinized_lat.Start() == fst::kNoStateId)
      KALDI_WARN << "Empty lattice while RescoreConstArpaLm (incompatible LM?)";

    // See lattice-lmrescore-const-arpa.cc
    fst::ArcSort(&determinized_lat, fst::OLabelCompare<kaldi::CompactLatticeArc>());
    kaldi::ConstArpaLmDeterministicFst const_arpa_fst(carpa_);
    kaldi::CompactLattice composed_clat;
    kaldi::ComposeCompactLatticeDeterministic(determinized_lat, &const_arpa_fst, &composed_clat);
    kaldi::Lattice composed_lat1;
    ConvertLattice(composed_clat, &composed_lat1);
    fst::Invert(&composed_lat1);
    DeterminizeLattice(composed_lat1, &clat);
}

void BaseNNet3OnlineModelWrapper::RescoreRnnlm(CompactLattice& clat, const std::string& prime_text) {
    ExecutionTimer timer("rnnlm rescoring");
    // See lattice-lmrescore-kaldi-rnnlm-pruned.cc
    rnnlm::KaldiRnnlmDeterministicFst lm_to_add_orig(rnnlm_max_ngram_order_, *rnnlm_info_);

    if (!prime_text.empty()) {
        istringstream iss(prime_text);
        vector<string> words{istream_iterator<string>{iss}, istream_iterator<string>{}};
        vector<int32> word_ids;
        word_ids.reserve(words.size());
        for (auto word : words) word_ids.push_back(word_syms_->Find(word));
        lm_to_add_orig.Prime(word_ids);
        KALDI_LOG << "RNNLM Primed with: \"" << prime_text << "\"";
    }

    DeterministicOnDemandFst<StdArc> *lm_to_add = new ScaleDeterministicOnDemandFst(rnnlm_scale_, &lm_to_add_orig);
    ComposeDeterministicOnDemandFst<StdArc> combined_lms(lm_to_subtract_det_scale_, lm_to_add);

    // Before composing with the LM FST, we scale the lattice weights
    // by the inverse of "lm_scale".  We'll later scale by "lm_scale".
    // We do it this way so we can determinize and it will give the
    // right effect (taking the "best path" through the LM) regardless
    // of the sign of lm_scale.
    // NOTE: The above comment is incorrect, but the below code is correct.
    if (decodable_config_.acoustic_scale != 1.0)
        ScaleLattice(AcousticLatticeScale(decodable_config_.acoustic_scale), &clat);
    TopSortCompactLatticeIfNeeded(&clat);

    // Composes lattice with language model.
    CompactLattice composed_clat;
    ComposeCompactLatticePruned(rnnlm_compose_opts_, clat, &combined_lms, &composed_clat);

    if (composed_clat.NumStates() == 0) {
        // Something went wrong. A warning will already have been printed.
        KALDI_WARN << "Empty lattice after RNNLM rescoring.";
        // FIXME: fall back to original?
    } else {
        if (decodable_config_.acoustic_scale != 1.0) {
            if (decodable_config_.acoustic_scale == 0.0)
                KALDI_ERR << "Acoustic scale cannot be zero.";
            ScaleLattice(AcousticLatticeScale(1.0 / decodable_config_.acoustic_scale), &composed_clat);
        }
        clat = composed_clat;
    }

    delete lm_to_add;
}

template <typename Decoder>
bool BaseNNet3OnlineModelWrapper::Decode(Decoder* decoder, BaseFloat samp_freq, const Vector<BaseFloat>& samples, bool finalize, bool save_adaptation_state) {
    ExecutionTimer timer("Decode", 2);

    if (!DecoderReady(decoder))
        KALDI_ERR << "Decoder not ready!";
    if (samp_freq != feature_info_->GetSamplingFrequency())
        KALDI_WARN << "Mismatched sampling frequency: " << samp_freq << " != " << feature_info_->GetSamplingFrequency() << " (model's)";

    if (samples.Dim() > 0) {
        feature_pipeline_->AcceptWaveform(samp_freq, samples);
        tot_frames_ += samples.Dim();
    }

    if (finalize)
        feature_pipeline_->InputFinished();  // No more input, so flush out last frames.

    if (silence_weighting_->Active()
            && feature_pipeline_->NumFramesReady() > 0
            && feature_pipeline_->IvectorFeature() != nullptr) {
        if (config_->silence_weight == 1.0)
            KALDI_WARN << "Computing silence weighting despite silence_weight == 1.0";
        std::vector<std::pair<int32, BaseFloat> > delta_weights;
        silence_weighting_->ComputeCurrentTraceback(decoder->Decoder());
        silence_weighting_->GetDeltaWeights(feature_pipeline_->NumFramesReady(), &delta_weights);  // FIXME: reuse decoder?
        feature_pipeline_->IvectorFeature()->UpdateFrameWeights(delta_weights);
    }

    decoder->AdvanceDecoding();

    if (finalize) {
        ExecutionTimer timer("Decode finalize", 2);
        decoder->FinalizeDecoding();
        decoder_finalized_ = true;

        tot_frames_decoded_ += tot_frames_;
        tot_frames_ = 0;

        if (save_adaptation_state)
            SaveAdaptationState();
    }

    return true;
}

template bool BaseNNet3OnlineModelWrapper::Decode(SingleUtteranceNnet3Decoder* decoder,
    BaseFloat samp_freq, const Vector<BaseFloat>& frames, bool finalize, bool save_adaptation_state);
template bool BaseNNet3OnlineModelWrapper::Decode(SingleUtteranceNnet3DecoderTpl<fst::ActiveGrammarFst>* decoder,
    BaseFloat samp_freq, const Vector<BaseFloat>& frames, bool finalize, bool save_adaptation_state);

} // namespace dragonfly


namespace kaldi {
    // Load IvectorExtractor config from JSON object. See src\online2\online-ivector-feature.h
    void from_json(const nlohmann::json& j, OnlineIvectorExtractionConfig& c) {
        if (!j.is_object()) KALDI_ERR << "Not an object!";
        for (auto& el : j.items()) {
            if (el.key() == "lda-matrix") j.at(el.key()).get_to(c.lda_mat_rxfilename);
            else if (el.key() == "global-cmvn-stats") j.at(el.key()).get_to(c.global_cmvn_stats_rxfilename);
            else if (el.key() == "cmvn-config") j.at(el.key()).get_to(c.cmvn_config_rxfilename);
            else if (el.key() == "online-cmvn-iextractor") j.at(el.key()).get_to(c.online_cmvn_iextractor);
            else if (el.key() == "splice-config") j.at(el.key()).get_to(c.splice_config_rxfilename);
            else if (el.key() == "diag-ubm") j.at(el.key()).get_to(c.diag_ubm_rxfilename);
            else if (el.key() == "ivector-extractor") j.at(el.key()).get_to(c.ivector_extractor_rxfilename);
            else if (el.key() == "ivector-period") j.at(el.key()).get_to(c.ivector_period);
            else if (el.key() == "num-gselect") j.at(el.key()).get_to(c.num_gselect);
            else if (el.key() == "min-post") j.at(el.key()).get_to(c.min_post);
            else if (el.key() == "posterior-scale") j.at(el.key()).get_to(c.posterior_scale);
            else if (el.key() == "max-count") j.at(el.key()).get_to(c.max_count);
            else if (el.key() == "use-most-recent-ivector") j.at(el.key()).get_to(c.use_most_recent_ivector);
            else if (el.key() == "greedy-ivector-extractor") j.at(el.key()).get_to(c.greedy_ivector_extractor);
            else if (el.key() == "max-remembered-frames") j.at(el.key()).get_to(c.max_remembered_frames);
            else KALDI_WARN << "unrecognized json object item " << el.key() << ": " << el.value();
        }
    }
} // namespace kaldi


extern "C" {
#include "dragonfly.h"
}

using namespace dragonfly;

bool load_lexicon_base_nnet3(void* model_vp, char* word_syms_filename_cp, char* word_align_lexicon_filename_cp) {
    try {
        auto model = static_cast<BaseNNet3OnlineModelWrapper*>(model_vp);
        std::string word_syms_filename(word_syms_filename_cp), word_align_lexicon_filename(word_align_lexicon_filename_cp);
        bool result = model->LoadLexicon(word_syms_filename, word_align_lexicon_filename);
        return result;

    } catch(const std::exception& e) {
        KALDI_WARN << "Trying to survive fatal exception: " << e.what();
        return false;
    }
}

bool save_adaptation_state_base_nnet3(void* model_vp) {
    try {
        auto model = static_cast<BaseNNet3OnlineModelWrapper*>(model_vp);
        bool result = model->SaveAdaptationState();
        return result;

    } catch(const std::exception& e) {
        KALDI_WARN << "Trying to survive fatal exception: " << e.what();
        return false;
    }
}

bool reset_adaptation_state_base_nnet3(void* model_vp) {
    try {
        auto model = static_cast<BaseNNet3OnlineModelWrapper*>(model_vp);
        model->ResetAdaptationState();
        return true;

    } catch(const std::exception& e) {
        KALDI_WARN << "Trying to survive fatal exception: " << e.what();
        return false;
    }
}

bool set_lm_prime_text_base_nnet3(void* model_vp, char* prime_text_cp) {
    try {
        auto model = static_cast<BaseNNet3OnlineModelWrapper*>(model_vp);
        std::string prime_text(prime_text_cp);
        model->SetLmPrimeText(prime_text);
        return true;

    } catch(const std::exception& e) {
        KALDI_WARN << "Trying to survive fatal exception: " << e.what();
        return false;
    }
}

bool get_word_align_base_nnet3(void* model_vp, int32_t* times_cp, int32_t* lengths_cp, int32_t num_words) {
    try {
        auto model = static_cast<BaseNNet3OnlineModelWrapper*>(model_vp);
        std::vector<string> words;
        std::vector<int32> times, lengths;
        bool result = model->GetWordAlignment(words, times, lengths, false);

        if (result) {
            KALDI_ASSERT(words.size() == num_words);
            for (size_t i = 0; i < words.size(); i++) {
                times_cp[i] = times[i];
                lengths_cp[i] = lengths[i];
            }
        } else {
            KALDI_WARN << "alignment failed";
        }

        return result;

    } catch(const std::exception& e) {
        KALDI_WARN << "Trying to survive fatal exception: " << e.what();
        return false;
    }
}

bool decode_base_nnet3(void* model_vp, float samp_freq, int32_t num_samples, float* samples, bool finalize, bool save_adaptation_state) {
    try {
        auto model = static_cast<BaseNNet3OnlineModelWrapper*>(model_vp);
        // if (num_samples > 3200)
        //     KALDI_WARN << "Decoding large block of " << num_samples << " samples!";
        Vector<BaseFloat> wave_data(num_samples, kUndefined);
        for (int i = 0; i < num_samples; i++)
            wave_data(i) = samples[i];
        bool result = model->Decode(samp_freq, wave_data, finalize, save_adaptation_state);
        return result;

    } catch(const std::exception& e) {
        KALDI_WARN << "Trying to survive fatal exception: " << e.what();
        return false;
    }
}

bool get_output_base_nnet3(void* model_vp, char* output, int32_t output_max_length,
        float* likelihood_p, float* am_score_p, float* lm_score_p, float* confidence_p, float* expected_error_rate_p) {
    try {
        auto model = static_cast<BaseNNet3OnlineModelWrapper*>(model_vp);
        if (output_max_length < 1) return false;
        std::string decoded_string;
        model->GetDecodedString(decoded_string, likelihood_p, am_score_p, lm_score_p, confidence_p, expected_error_rate_p);

        // KALDI_LOG << "sleeping";
        // std::this_thread::sleep_for(std::chrono::milliseconds(25));
        // KALDI_LOG << "slept";

        const char* cstr = decoded_string.c_str();
        strncpy(output, cstr, output_max_length);
        output[output_max_length - 1] = 0;
        return true;

    } catch(const std::exception& e) {
        KALDI_WARN << "Trying to survive fatal exception: " << e.what();
        return false;
    }
}
