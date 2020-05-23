// Defines the exported functions for the DLL application.

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

extern "C" {
#include "dragonfly.h"
}

#include "feat/wave-reader.h"
#include "online2/online-feature-pipeline.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/compose-lattice-pruned.h"
#include "lat/lattice-functions.h"
#include "lat/word-align-lattice-lexicon.h"
#include "rnnlm/rnnlm-lattice-rescoring.h"
#include "nnet3/nnet-utils.h"

#define DEFAULT_VERBOSITY 1

namespace dragonfly {
    using namespace kaldi;
    using namespace fst;

    // ConstFst<StdArc>* CastOrConvertToConstFst(Fst<StdArc>* fst) {
    //     // This version currently supports ConstFst<StdArc> or VectorFst<StdArc>
    //     std::string real_type = fst->Type();
    //     KALDI_ASSERT(real_type == "vector" || real_type == "const");
    //     if (real_type == "const") {
    //         return dynamic_cast<ConstFst<StdArc>*>(fst);
    //     } else {
    //         // As the 'fst' can't cast to ConstFst, we carete a new
    //         // ConstFst<StdArc> initialized by 'fst', and delete 'fst'.
    //         ConstFst<StdArc>* new_fst = new ConstFst<StdArc>(*fst);
    //         delete fst;
    //         return new_fst;
    //     }
    // }

    class PlainNNet3OnlineModelWrapper {
    public:

        PlainNNet3OnlineModelWrapper(
            BaseFloat beam, int32 max_active, int32 min_active, BaseFloat lattice_beam, BaseFloat acoustic_scale, int32 frame_subsampling_factor,
            std::string& mfcc_config_filename, std::string& ie_config_filename, std::string& model_filename,
            std::string& word_syms_filename, std::string& word_align_lexicon_filename, std::string& fst_filename,
            int32 verbosity = DEFAULT_VERBOSITY);
        ~PlainNNet3OnlineModelWrapper();

        bool LoadLexicon(std::string& word_syms_filename, std::string& word_align_lexicon_filename);
        void ResetAdaptationState();
        bool Decode(BaseFloat samp_freq, int32 num_frames, BaseFloat* frames, bool finalize, bool save_adaptation_state = true);

        void GetDecodedString(std::string& decoded_string, double& likelihood);
        bool GetWordAlignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths, bool include_eps);

    protected:

        // Model
        fst::SymbolTable *word_syms = nullptr;
        std::vector<std::vector<int32> > word_align_lexicon;  // for each word, its word-id + word-id + a list of its phones
        StdConstFst* decode_fst = nullptr;

        // Model objects
        OnlineNnet2FeaturePipelineConfig feature_config;
        nnet3::NnetSimpleLoopedComputationOptions decodable_config;
        LatticeFasterDecoderConfig decoder_config;
        OnlineEndpointConfig endpoint_config;
        TransitionModel trans_model;
        nnet3::AmNnetSimple am_nnet;
        OnlineNnet2FeaturePipelineInfo* feature_info = nullptr;
        nnet3::DecodableNnetSimpleLoopedInfo* decodable_info = nullptr;  // contains precomputed stuff that is used by all decodable objects

        // Decoder objects
        OnlineNnet2FeaturePipeline* feature_pipeline = nullptr;
        OnlineSilenceWeighting* silence_weighting = nullptr;  // reinstantiated per utterance
        OnlineIvectorExtractorAdaptationState* adaptation_state = nullptr;
        SingleUtteranceNnet3Decoder* decoder = nullptr;
        int32 tot_frames, tot_frames_decoded;
        CompactLattice best_path_clat;
        WordAlignLatticeLexiconInfo* word_align_lexicon_info = nullptr;
        std::set<int32> word_align_lexicon_words;  // contains word-ids that are in word_align_lexicon_info
        bool best_path_has_valid_word_align;

        bool enable_rnnlm_ = true;
        nnet3::Nnet rnnlm_;
        CuMatrix<BaseFloat> word_embedding_mat_;
        fst::ScaleDeterministicOnDemandFst* lm_to_subtract_det_scale_ = nullptr;
        rnnlm::RnnlmComputeStateComputationOptions rnnlm_opts_;
        rnnlm::RnnlmComputeStateInfo* rnnlm_info_ = nullptr;
        int32 rnnlm_max_ngram_order_;
        ComposeLatticePrunedOptions rnnlm_compose_opts_;
        BaseFloat rnnlm_scale_;

        void StartDecoding();
        void FreeDecoder(void);
    };

    PlainNNet3OnlineModelWrapper::PlainNNet3OnlineModelWrapper(
        BaseFloat beam, int32 max_active, int32 min_active, BaseFloat lattice_beam, BaseFloat acoustic_scale, int32 frame_subsampling_factor,
        std::string& mfcc_config_filename, std::string& ie_config_filename, std::string& model_filename,
        std::string& word_syms_filename, std::string& word_align_lexicon_filename, std::string& fst_filename,
        int32 verbosity) {
        if (verbosity >= 2) {
            KALDI_LOG << "word_syms_filename: " << word_syms_filename;
            KALDI_LOG << "word_align_lexicon_filename: " << word_align_lexicon_filename;
            KALDI_LOG << "mfcc_config_filename: " << mfcc_config_filename;
            KALDI_LOG << "ie_config_filename: " << ie_config_filename;
            KALDI_LOG << "model_filename: " << model_filename;
            KALDI_LOG << "fst_filename: " << fst_filename;
        } else if (verbosity == 1) {
            SetLogHandler([](const LogMessageEnvelope& envelope, const char* message) {
                if (envelope.severity <= LogMessageEnvelope::kWarning) {
                    std::cerr << "[KALDI severity=" << envelope.severity << "] " << message << "\n";
                }
            });
        } else {
            // Silence kaldi output as well
            SetLogHandler([](const LogMessageEnvelope& envelope, const char* message) {});
        }

        ParseOptions po("");
        feature_config.Register(&po);
        decodable_config.Register(&po);
        decoder_config.Register(&po);
        endpoint_config.Register(&po);

        feature_config.mfcc_config = mfcc_config_filename;
        feature_config.ivector_extraction_config = ie_config_filename;
        decoder_config.max_active = max_active;
        decoder_config.min_active = min_active;
        decoder_config.beam = beam;
        decoder_config.lattice_beam = lattice_beam;
        decodable_config.acoustic_scale = acoustic_scale;
        decodable_config.frame_subsampling_factor = frame_subsampling_factor;

        {
            bool binary;
            Input ki(model_filename, &binary);
            trans_model.Read(ki.Stream(), binary);
            am_nnet.Read(ki.Stream(), binary);
            SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
            SetDropoutTestMode(true, &(am_nnet.GetNnet()));
            nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
        }

        feature_info = new OnlineNnet2FeaturePipelineInfo(feature_config);
        decodable_info = new nnet3::DecodableNnetSimpleLoopedInfo(decodable_config, &am_nnet);
        ResetAdaptationState();

        decode_fst = dynamic_cast<StdConstFst*>(ReadFstKaldiGeneric(fst_filename));

        LoadLexicon(word_syms_filename, word_align_lexicon_filename);

        if (enable_rnnlm_) {
            std::string rnnlm_dir = "../../../../CompCom-edge/kaldi_model_daanzu.standard1-unk/";
            // std::string rnnlm_dir = std::filesystem::path(model_filename).replace_filename("rnnlm");
            VectorFst<StdArc> *lm_to_subtract_fst = ReadAndPrepareLmFst(rnnlm_dir + "G.fst");
            BackoffDeterministicOnDemandFst<StdArc> *lm_to_subtract_det_backoff = new BackoffDeterministicOnDemandFst<StdArc>(*lm_to_subtract_fst);
            rnnlm_scale_ = 1.0;
            lm_to_subtract_det_scale_ = new ScaleDeterministicOnDemandFst(-rnnlm_scale_, lm_to_subtract_det_backoff);

            ReadKaldiObject(rnnlm_dir + "rnnlm/" + "final.raw", &rnnlm_);
            KALDI_ASSERT(IsSimpleNnet(rnnlm_));
            ReadKaldiObject(rnnlm_dir + "rnnlm/" + "word_embedding.mat", &word_embedding_mat_);

            rnnlm_opts_.bos_index = word_syms->Find("<s>");
            rnnlm_opts_.eos_index = word_syms->Find("</s>");
            rnnlm_info_ = new rnnlm::RnnlmComputeStateInfo(rnnlm_opts_, rnnlm_, word_embedding_mat_);
            rnnlm_max_ngram_order_ = 4;
        }

        decoder = nullptr;
        tot_frames = 0;
        tot_frames_decoded = 0;
    }

    PlainNNet3OnlineModelWrapper::~PlainNNet3OnlineModelWrapper() {
        FreeDecoder();
        delete feature_info;
        delete decodable_info;
        if (word_align_lexicon_info)
            delete word_align_lexicon_info;
    }

    bool PlainNNet3OnlineModelWrapper::LoadLexicon(std::string& word_syms_filename, std::string& word_align_lexicon_filename) {
        // FIXME: make more robust to errors
        
        if (word_syms_filename != "") {
            if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename))) {
                KALDI_ERR << "Could not read symbol table from file " << word_syms_filename;
                return false;
            }
        }

        if (word_align_lexicon_filename != "") {
        	bool binary_in;
        	Input ki(word_align_lexicon_filename, &binary_in);
        	KALDI_ASSERT(!binary_in && "Not expecting binary file for lexicon");
        	if (!ReadLexiconForWordAlign(ki.Stream(), &word_align_lexicon)) {
        		KALDI_ERR << "Error reading word alignment lexicon from file " << word_align_lexicon_filename;
                return false;
        	}
            if (word_align_lexicon_info)
                delete word_align_lexicon_info;
            word_align_lexicon_info = new WordAlignLatticeLexiconInfo(word_align_lexicon);

            word_align_lexicon_words.clear();
            for (auto entry : word_align_lexicon)
                word_align_lexicon_words.insert(entry.at(0));
        }

        return true;
    }

    void PlainNNet3OnlineModelWrapper::ResetAdaptationState() {
        // NOTE: assumes single speaker; optionally maintains adaptation state
        if (adaptation_state != nullptr) {
            delete adaptation_state;
        }
        adaptation_state = new OnlineIvectorExtractorAdaptationState(feature_info->ivector_extractor_info);
    }

    void PlainNNet3OnlineModelWrapper::StartDecoding() {
        FreeDecoder();
        feature_pipeline = new OnlineNnet2FeaturePipeline(*feature_info);
        feature_pipeline->SetAdaptationState(*adaptation_state);
        silence_weighting = new OnlineSilenceWeighting(
            trans_model, feature_info->silence_weighting_config,
            decodable_config.frame_subsampling_factor);
        decoder = new SingleUtteranceNnet3Decoder(
            decoder_config, trans_model, *decodable_info, *decode_fst, feature_pipeline);
        best_path_has_valid_word_align = false;
    }

    void PlainNNet3OnlineModelWrapper::FreeDecoder(void) {
        if (decoder) {
            delete decoder;
            decoder = nullptr;
        }
        if (silence_weighting) {
            delete silence_weighting;
            silence_weighting = nullptr;
        }
        if (feature_pipeline) {
            delete feature_pipeline;
            feature_pipeline = nullptr;
        }
    }

    // grammars_activity is ignored once decoding has already started
    bool PlainNNet3OnlineModelWrapper::Decode(BaseFloat samp_freq, int32 num_frames, BaseFloat* frames, bool finalize, bool save_adaptation_state) {
        using fst::VectorFst;

        if (!decoder)
            StartDecoding();
        //else if (grammars_activity.size() != 0)
        //	KALDI_WARN << "non-empty grammars_activity passed on already-started decode";

        Vector<BaseFloat> wave_part(num_frames, kUndefined);
        for (int i = 0; i<num_frames; i++) {
            wave_part(i) = frames[i];
        }
        tot_frames += num_frames;

        feature_pipeline->AcceptWaveform(samp_freq, wave_part);

        if (finalize) {
            // no more input; flush out last frames
            feature_pipeline->InputFinished();
        }

        if (silence_weighting->Active()
                && feature_pipeline->NumFramesReady() > 0
                && feature_pipeline->IvectorFeature() != nullptr) {
            std::vector<std::pair<int32, BaseFloat> > delta_weights;
            silence_weighting->ComputeCurrentTraceback(decoder->Decoder());
            silence_weighting->GetDeltaWeights(feature_pipeline->NumFramesReady(), &delta_weights);
            feature_pipeline->IvectorFeature()->UpdateFrameWeights(delta_weights);
        }

        decoder->AdvanceDecoding();

        if (finalize) {
            decoder->FinalizeDecoding();

            CompactLattice clat;
            bool end_of_utterance = true;
            decoder->GetLattice(end_of_utterance, &clat);

            if (clat.NumStates() == 0) {
                KALDI_WARN << "Empty lattice.";
                return false;
            }

            if (enable_rnnlm_) {
                rnnlm::KaldiRnnlmDeterministicFst lm_to_add_orig(rnnlm_max_ngram_order_, *rnnlm_info_);
                // std::vector<int32> precontext({(int32)word_syms->Find("invoice")});
                // std::vector<int32> precontext(10, word_syms->Find("invoice"));
                // lm_to_add_orig.Prime(precontext);
                DeterministicOnDemandFst<StdArc> *lm_to_add = new ScaleDeterministicOnDemandFst(rnnlm_scale_, &lm_to_add_orig);

                // Before composing with the LM FST, we scale the lattice weights
                // by the inverse of "lm_scale".  We'll later scale by "lm_scale".
                // We do it this way so we can determinize and it will give the
                // right effect (taking the "best path" through the LM) regardless
                // of the sign of lm_scale.
                // NOTE: The above comment is incorrect, but the below code is correct.
                if (decodable_config.acoustic_scale != 1.0)
                    ScaleLattice(AcousticLatticeScale(decodable_config.acoustic_scale), &clat);
                TopSortCompactLatticeIfNeeded(&clat);

                ComposeDeterministicOnDemandFst<StdArc> combined_lms(lm_to_subtract_det_scale_, lm_to_add);

                // Composes lattice with language model.
                CompactLattice composed_clat;
                ComposeCompactLatticePruned(rnnlm_compose_opts_, clat, &combined_lms, &composed_clat);

                if (composed_clat.NumStates() == 0) {
                    // Something went wrong.  A warning will already have been printed.
                    KALDI_WARN << "Empty lattice after RNNLM rescoring.";
                    // FIXME: fall back to original?
                } else {
                    if (decodable_config.acoustic_scale != 1.0) {
                        if (decodable_config.acoustic_scale == 0.0)
                            KALDI_ERR << "Acoustic scale cannot be zero.";
                        ScaleLattice(AcousticLatticeScale(1.0 / decodable_config.acoustic_scale), &composed_clat);
                    }
                    clat = composed_clat;
                }

                delete lm_to_add;
            }

            CompactLatticeShortestPath(clat, &best_path_clat);

            // BaseFloat inv_acoustic_scale = 1.0 / decodable_config.acoustic_scale;
            // ScaleLattice(AcousticLatticeScale(inv_acoustic_scale), &clat);

            // FIXME: decide whether to save adaptation?
            if (save_adaptation_state) {
                feature_pipeline->GetAdaptationState(adaptation_state);
                KALDI_LOG << "Saved adaptation state.";
            }

            tot_frames_decoded = tot_frames;
            tot_frames = 0;

            FreeDecoder();
        }

        return true;
    }

    void PlainNNet3OnlineModelWrapper::GetDecodedString(std::string& decoded_string, double& likelihood) {
        Lattice best_path_lat;

        if (decoder) {
            // Decoding is not finished yet, so we will look up the best partial result so far

            if (decoder->NumFramesDecoded() == 0) {
                likelihood = 0.0;
                return;
            }

            decoder->GetBestPath(false, &best_path_lat);
        } else {
            ConvertLattice(best_path_clat, &best_path_lat);
        }

        std::vector<int32> words;
        std::vector<int32> alignment;
        LatticeWeight weight;
        int32 num_frames;
        GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
        num_frames = alignment.size();
        likelihood = -(weight.Value1() + weight.Value2()) / num_frames;

        decoded_string = "";
        best_path_has_valid_word_align = true;
        for (size_t i = 0; i < words.size(); i++) {
            std::string s = word_syms->Find(words[i]);
            if (i != 0)
                decoded_string += ' ';
            if (s == "") {
                KALDI_WARN << "Word-id " << words[i] << " not in symbol table";
                s = "MISSING_WORD";
            }
            decoded_string += s;
            if (!word_align_lexicon_words.count(words[i]))
                best_path_has_valid_word_align = false;
        }
    }

    bool PlainNNet3OnlineModelWrapper::GetWordAlignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths, bool include_eps) {
        if (!word_align_lexicon.size() || !word_align_lexicon_info) {
            KALDI_WARN << "No word alignment lexicon loaded";
            return false;
        }

        if (!best_path_has_valid_word_align) {
            KALDI_WARN << "Word not in word alignment lexicon";
            return false;
        }

        CompactLattice aligned_clat;
        WordAlignLatticeLexiconOpts opts;
        bool ok = WordAlignLatticeLexicon(best_path_clat, trans_model, *word_align_lexicon_info, opts, &aligned_clat);

        if (!ok) {
            KALDI_WARN << "Lattice did not align correctly";
            return false;

        } else {
            if (aligned_clat.Start() == fst::kNoStateId) {
                KALDI_WARN << "Lattice was empty";
                return false;

            } else {
                TopSortCompactLatticeIfNeeded(&aligned_clat);

                // lattice-1best
                CompactLattice best_path_aligned;
                CompactLatticeShortestPath(aligned_clat, &best_path_aligned); 

                // nbest-to-ctm
                std::vector<int32> word_idxs, times_raw, lengths_raw;
                if (!CompactLatticeToWordAlignment(best_path_aligned, &word_idxs, &times_raw, &lengths_raw)) {
                    KALDI_WARN << "CompactLatticeToWordAlignment failed.";
                    return false;
                }

                // lexicon lookup
                words.clear();
                for (size_t i = 0; i < word_idxs.size(); i++) {
                    std::string s = word_syms->Find(word_idxs[i]);  // Must be found, or CompactLatticeToWordAlignment would have crashed
                    // KALDI_LOG << "align: " << s << " - " << times_raw[i] << " - " << lengths_raw[i];
                    if (include_eps || (word_idxs[i] != 0)) {
                        words.push_back(s);
                        times.push_back(times_raw[i]);
                        lengths.push_back(lengths_raw[i]);
                    }
                }
                return true;
            }
        }
    }
}

using namespace dragonfly;

void* init_plain_nnet3(float beam, int32_t max_active, int32_t min_active, float lattice_beam, float acoustic_scale, int32_t frame_subsampling_factor,
    char* mfcc_config_filename_cp, char* ie_config_filename_cp, char* model_filename_cp,
    char* word_syms_filename_cp, char* word_align_lexicon_filename_cp, char* fst_filename_cp,
    int32_t verbosity) {
    std::string word_syms_filename(word_syms_filename_cp),
        word_align_lexicon_filename((word_align_lexicon_filename_cp != nullptr) ? word_align_lexicon_filename_cp : ""),
        mfcc_config_filename(mfcc_config_filename_cp),
        ie_config_filename(ie_config_filename_cp),
        model_filename(model_filename_cp),
        fst_filename(fst_filename_cp);
    PlainNNet3OnlineModelWrapper* model = new PlainNNet3OnlineModelWrapper(beam, max_active, min_active, lattice_beam, acoustic_scale, frame_subsampling_factor,
        mfcc_config_filename, ie_config_filename, model_filename,
        word_syms_filename, word_align_lexicon_filename, fst_filename,
        verbosity);
    return model;
}

bool decode_plain_nnet3(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize, bool save_adaptation_state) {
    try {
        PlainNNet3OnlineModelWrapper* model = static_cast<PlainNNet3OnlineModelWrapper*>(model_vp);
        bool result = model->Decode(samp_freq, num_frames, frames, finalize, save_adaptation_state);
        return result;

    } catch(const std::exception& e) {
        KALDI_WARN << "Trying to survive fatal exception: " << e.what();
        return false;
    }
}

bool reset_adaptation_state_plain_nnet3(void* model_vp) {
    try {
        PlainNNet3OnlineModelWrapper* model = static_cast<PlainNNet3OnlineModelWrapper*>(model_vp);
        model->ResetAdaptationState();
        return true;

    } catch(const std::exception& e) {
        KALDI_WARN << "Trying to survive fatal exception: " << e.what();
        return false;
    }
}

bool get_output_plain_nnet3(void* model_vp, char* output, int32_t output_max_length, double* likelihood_p) {
    try {
        if (output_max_length < 1) return false;
        PlainNNet3OnlineModelWrapper* model = static_cast<PlainNNet3OnlineModelWrapper*>(model_vp);
        std::string decoded_string;
        double likelihood;
        model->GetDecodedString(decoded_string, likelihood);
        const char* cstr = decoded_string.c_str();
        strncpy(output, cstr, output_max_length);
        output[output_max_length - 1] = 0;
        *likelihood_p = likelihood;
        return true;

    } catch(const std::exception& e) {
        KALDI_WARN << "Trying to survive fatal exception: " << e.what();
        return false;
    }
}

bool get_word_align_plain_nnet3(void* model_vp, int32_t* times_cp, int32_t* lengths_cp, int32_t num_words) {
    try {
        PlainNNet3OnlineModelWrapper* model = static_cast<PlainNNet3OnlineModelWrapper*>(model_vp);
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
