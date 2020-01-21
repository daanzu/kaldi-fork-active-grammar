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
#include "lat/lattice-functions.h"
#include "lat/word-align-lattice-lexicon.h"
#include "nnet3/nnet-utils.h"
#include "decoder/active-grammar-fst.h"

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

    class AgfNNet3OnlineModelWrapper {
    public:

        AgfNNet3OnlineModelWrapper(
            BaseFloat beam, int32 max_active, int32 min_active, BaseFloat lattice_beam, BaseFloat acoustic_scale, int32 frame_subsampling_factor,
            std::string& mfcc_config_filename, std::string& ie_config_filename, std::string& model_filename,
            int32 nonterm_phones_offset, int32 rules_nonterm_offset, int32 dictation_nonterm_offset,
            std::string& word_syms_filename, std::string& word_align_lexicon_filename,
            std::string& top_fst_filename, std::string& dictation_fst_filename,
            int32 verbosity = DEFAULT_VERBOSITY);
        ~AgfNNet3OnlineModelWrapper();

        bool load_lexicon(std::string& word_syms_filename, std::string& word_align_lexicon_filename);
        int32 add_grammar_fst(std::string& grammar_fst_filename);
        bool reload_grammar_fst(int32 grammar_fst_index, std::string& grammar_fst_filename);
        bool remove_grammar_fst(int32 grammar_fst_index);
        bool save_adaptation_state();
        void reset_adaptation_state();
        bool decode(BaseFloat samp_freq, int32 num_frames, BaseFloat* frames, bool finalize, std::vector<bool>& grammars_activity, bool save_adaptation_state = true);

        void get_decoded_string(std::string& decoded_string, double& likelihood);
        bool get_word_alignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths, bool include_eps);

    protected:

        // Model
        int32 nonterm_phones_offset;
        int32 dictation_phones_offset;  // offset from nonterm_phones_offset that the dictation nonterms are
        int32 rules_phones_offset;  // offset from nonterm_phones_offset that the kaldi_rules nonterms are
        fst::SymbolTable *word_syms = nullptr;
        std::vector<std::vector<int32> > word_align_lexicon;  // for each word, its word-id + word-id + a list of its phones
        StdConstFst *top_fst = nullptr;
        StdConstFst *dictation_fst = nullptr;
        std::vector<StdConstFst*> grammar_fsts;
        std::map<StdConstFst*, std::string> grammar_fsts_filename_map;  // maps grammar_fst -> name; for debugging
        // INVARIANT: same size: grammar_fsts, grammar_fsts_filename_map, grammar_fsts_enabled

        // Model objects
        OnlineNnet2FeaturePipelineConfig feature_config;
        nnet3::NnetSimpleLoopedComputationOptions decodable_config;
        LatticeFasterDecoderConfig decoder_config;
        OnlineEndpointConfig endpoint_config;
        TransitionModel trans_model;
        nnet3::AmNnetSimple am_nnet;
        OnlineNnet2FeaturePipelineInfo* feature_info = nullptr;
        nnet3::DecodableNnetSimpleLoopedInfo* decodable_info = nullptr;  // contains precomputed stuff that is used by all decodable objects
        ActiveGrammarFst* active_grammar_fst = nullptr;

        // Decoder objects
        OnlineNnet2FeaturePipeline* feature_pipeline = nullptr;
        OnlineSilenceWeighting* silence_weighting = nullptr;  // reinstantiated per utterance
        OnlineIvectorExtractorAdaptationState* adaptation_state = nullptr;
        SingleUtteranceNnet3DecoderTpl<fst::ActiveGrammarFst>* decoder = nullptr;
        int32 tot_frames, tot_frames_decoded;
        CompactLattice best_path_clat;
        WordAlignLatticeLexiconInfo* word_align_lexicon_info = nullptr;
        std::set<int32> word_align_lexicon_words;  // contains word-ids that are in word_align_lexicon_info
        bool best_path_has_valid_word_align;

        StdConstFst* read_fst_file(std::string filename);
        void start_decoding(std::vector<bool> grammars_activity);
        void free_decoder(bool keep_feature_pipeline = false);
    };

    AgfNNet3OnlineModelWrapper::AgfNNet3OnlineModelWrapper(
        BaseFloat beam, int32 max_active, int32 min_active, BaseFloat lattice_beam, BaseFloat acoustic_scale, int32 frame_subsampling_factor,
        std::string& mfcc_config_filename, std::string& ie_config_filename, std::string& model_filename,
        int32 nonterm_phones_offset, int32 rules_nonterm_offset, int32 dictation_nonterm_offset,
        std::string& word_syms_filename, std::string& word_align_lexicon_filename,
        std::string& top_fst_filename, std::string& dictation_fst_filename,
        int32 verbosity) {
        if (verbosity >= 2) {
            KALDI_LOG << "nonterm_phones_offset: " << nonterm_phones_offset;
            KALDI_LOG << "rules_nonterm_offset: " << rules_nonterm_offset;
            KALDI_LOG << "dictation_nonterm_offset: " << dictation_nonterm_offset;
            KALDI_LOG << "word_syms_filename: " << word_syms_filename;
            KALDI_LOG << "word_align_lexicon_filename: " << word_align_lexicon_filename;
            KALDI_LOG << "mfcc_config_filename: " << mfcc_config_filename;
            KALDI_LOG << "ie_config_filename: " << ie_config_filename;
            KALDI_LOG << "model_filename: " << model_filename;
            KALDI_LOG << "top_fst_filename: " << top_fst_filename;
            KALDI_LOG << "dictation_fst_filename: " << dictation_fst_filename;
            KALDI_LOG << "kNontermBigNumber, GetEncodingMultiple: " << kNontermBigNumber << ", " << GetEncodingMultiple(nonterm_phones_offset);
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
        reset_adaptation_state();
        top_fst = dynamic_cast<StdConstFst*>(ReadFstKaldiGeneric(top_fst_filename));

        this->nonterm_phones_offset = nonterm_phones_offset;
        rules_phones_offset = nonterm_phones_offset + rules_nonterm_offset;
        if (!dictation_fst_filename.empty()) {
            dictation_phones_offset = nonterm_phones_offset + dictation_nonterm_offset;
            dictation_fst = read_fst_file(dictation_fst_filename);
        } else {
            dictation_phones_offset = 0;
            dictation_fst = nullptr;
        }

        load_lexicon(word_syms_filename, word_align_lexicon_filename);

        active_grammar_fst = nullptr;
        decoder = nullptr;
        tot_frames = 0;
        tot_frames_decoded = 0;
    }

    AgfNNet3OnlineModelWrapper::~AgfNNet3OnlineModelWrapper() {
        free_decoder();
        delete feature_info;
        delete decodable_info;
        if (word_align_lexicon_info)
            delete word_align_lexicon_info;
    }

    bool AgfNNet3OnlineModelWrapper::load_lexicon(std::string& word_syms_filename, std::string& word_align_lexicon_filename) {
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

    StdConstFst* AgfNNet3OnlineModelWrapper::read_fst_file(std::string filename) {
        if (filename.compare(filename.length() - 4, 4, ".txt") == 0) {
            // FIXME: fstdeterminize | fstminimize | fstrmepsilon | fstarcsort --sort_type=ilabel
            KALDI_WARN << "cannot read text fst file " << filename;
            return nullptr;
        } else {
            return dynamic_cast<StdConstFst*>(ReadFstKaldiGeneric(filename));
        }
    }

    int32 AgfNNet3OnlineModelWrapper::add_grammar_fst(std::string& grammar_fst_filename) {
        auto grammar_fst_index = grammar_fsts.size();
        auto grammar_fst = read_fst_file(grammar_fst_filename);
        KALDI_LOG << "adding FST #" << grammar_fst_index << " @ 0x" << grammar_fst << " " << grammar_fst_filename;
        grammar_fsts.emplace_back(grammar_fst);
        grammar_fsts_filename_map[grammar_fst] = grammar_fst_filename;
        if (active_grammar_fst) {
            delete active_grammar_fst;
            active_grammar_fst = nullptr;
        }
        return grammar_fst_index;
    }

    bool AgfNNet3OnlineModelWrapper::reload_grammar_fst(int32 grammar_fst_index, std::string& grammar_fst_filename) {
        auto old_grammar_fst = grammar_fsts.at(grammar_fst_index);
        grammar_fsts_filename_map.erase(old_grammar_fst);
        delete old_grammar_fst;

        auto grammar_fst = read_fst_file(grammar_fst_filename);
        KALDI_LOG << "reloading FST #" << grammar_fst_index << " @ 0x" << grammar_fst << " " << grammar_fst_filename;
        grammar_fsts.at(grammar_fst_index) = grammar_fst;
        grammar_fsts_filename_map[grammar_fst] = grammar_fst_filename;
        if (active_grammar_fst) {
            delete active_grammar_fst;
            active_grammar_fst = nullptr;
        }
        return true;
    }

    bool AgfNNet3OnlineModelWrapper::remove_grammar_fst(int32 grammar_fst_index) {
        auto grammar_fst = grammar_fsts.at(grammar_fst_index);
        KALDI_LOG << "removing FST #" << grammar_fst_index << " @ 0x" << grammar_fst << " " << grammar_fsts_filename_map.at(grammar_fst);
        grammar_fsts.erase(grammar_fsts.begin() + grammar_fst_index);
        grammar_fsts_filename_map.erase(grammar_fst);
        delete grammar_fst;
        if (active_grammar_fst) {
            delete active_grammar_fst;
            active_grammar_fst = nullptr;
        }
        return true;
    }

    bool AgfNNet3OnlineModelWrapper::save_adaptation_state() {
        if (feature_pipeline != nullptr) {
            feature_pipeline->GetAdaptationState(adaptation_state);
            KALDI_LOG << "Saved adaptation state.";
            return true;
        }
        return false;
    }

    void AgfNNet3OnlineModelWrapper::reset_adaptation_state() {
        if (adaptation_state != nullptr) {
            delete adaptation_state;
        }
        adaptation_state = new OnlineIvectorExtractorAdaptationState(feature_info->ivector_extractor_info);
    }

    void AgfNNet3OnlineModelWrapper::start_decoding(std::vector<bool> grammars_activity) {
        free_decoder();

        if (active_grammar_fst == nullptr) {
            // Timer timer(true);
            std::vector<std::pair<int32, const StdConstFst *> > ifsts;
            for (auto grammar_fst : grammar_fsts) {
                int32 nonterm_phone = rules_phones_offset + ifsts.size();
                ifsts.emplace_back(std::make_pair(nonterm_phone, grammar_fst));
            }
            if (dictation_fst != nullptr) {
                ifsts.emplace_back(std::make_pair(dictation_phones_offset, dictation_fst));
            }
            active_grammar_fst = new ActiveGrammarFst(nonterm_phones_offset, *top_fst, ifsts);
            // KALDI_LOG << "built new ActiveGrammarFst" << " in " << (timer.Elapsed() * 1000) << "ms.";
        }
        grammars_activity.emplace_back(dictation_fst != nullptr);  // dictation_fst is only enabled if present
        active_grammar_fst->UpdateActivity(grammars_activity);

        feature_pipeline = new OnlineNnet2FeaturePipeline(*feature_info);
        feature_pipeline->SetAdaptationState(*adaptation_state);
        silence_weighting = new OnlineSilenceWeighting(
            trans_model, feature_info->silence_weighting_config,
            decodable_config.frame_subsampling_factor);
        decoder = new SingleUtteranceNnet3DecoderTpl<fst::ActiveGrammarFst>(
            decoder_config, trans_model, *decodable_info, *active_grammar_fst, feature_pipeline);
        best_path_has_valid_word_align = false;
    }

    void AgfNNet3OnlineModelWrapper::free_decoder(bool keep_feature_pipeline) {
        if (decoder) {
            delete decoder;
            decoder = nullptr;
        }
        if (silence_weighting) {
            delete silence_weighting;
            silence_weighting = nullptr;
        }
        if (feature_pipeline && !keep_feature_pipeline) {
            delete feature_pipeline;
            feature_pipeline = nullptr;
        }
    }

    // grammars_activity is ignored once decoding has already started
    bool AgfNNet3OnlineModelWrapper::decode(BaseFloat samp_freq, int32 num_frames, BaseFloat* frames, bool finalize,
        std::vector<bool>& grammars_activity, bool save_adaptation_state) {
        using fst::VectorFst;

        if (!decoder)
            start_decoding(grammars_activity);
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

            CompactLatticeShortestPath(clat, &best_path_clat);

            // BaseFloat inv_acoustic_scale = 1.0 / decodable_config.acoustic_scale;
            // ScaleLattice(AcousticLatticeScale(inv_acoustic_scale), &clat);

            tot_frames_decoded = tot_frames;
            tot_frames = 0;

            free_decoder(true);

            if (save_adaptation_state) {
                feature_pipeline->GetAdaptationState(adaptation_state);
                KALDI_LOG << "Saved adaptation state.";
                free_decoder();
                // std::string output;
                // double likelihood;
                // get_decoded_string(output, likelihood);
                // // int count_terminals = std::count_if(output.begin(), output.end(), [](std::string word){ return word[0] != '#'; });
                // if (output.size() > 0) {
                //     feature_pipeline->GetAdaptationState(adaptation_state);
                //     KALDI_LOG << "Saved adaptation state." << output;
                //     free_decoder();
                // } else {
                //     KALDI_LOG << "Did not save adaptation state, because empty recognition.";
                // }
            }
        }

        return true;
    }

    void AgfNNet3OnlineModelWrapper::get_decoded_string(std::string& decoded_string, double& likelihood) {
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
            if (!word_align_lexicon_words.count(words[i])) {
                best_path_has_valid_word_align = false;
                KALDI_LOG << "Word " << s << " (id #" << words[i] << ") not in word alignment lexicon";
            }
        }
    }

    bool AgfNNet3OnlineModelWrapper::get_word_alignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths, bool include_eps) {
        if (!word_align_lexicon.size() || !word_align_lexicon_info) {
            KALDI_WARN << "No word alignment lexicon loaded";
            return false;
        }

        if (!best_path_has_valid_word_align) {
            KALDI_WARN << "There was a word not in word alignment lexicon";
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

void* init_agf_nnet3(float beam, int32_t max_active, int32_t min_active, float lattice_beam, float acoustic_scale, int32_t frame_subsampling_factor,
    char* mfcc_config_filename_cp, char* ie_config_filename_cp, char* model_filename_cp,
    int32_t nonterm_phones_offset, int32_t rules_nonterm_offset, int32_t dictation_nonterm_offset,
    char* word_syms_filename_cp, char* word_align_lexicon_filename_cp,
    char* top_fst_filename_cp, char* dictation_fst_filename_cp,
    int32_t verbosity) {
    std::string word_syms_filename(word_syms_filename_cp),
        word_align_lexicon_filename((word_align_lexicon_filename_cp != nullptr) ? word_align_lexicon_filename_cp : ""),
        mfcc_config_filename(mfcc_config_filename_cp),
        ie_config_filename(ie_config_filename_cp),
        model_filename(model_filename_cp),
        top_fst_filename(top_fst_filename_cp),
        dictation_fst_filename((dictation_fst_filename_cp != nullptr) ? dictation_fst_filename_cp : "");
    AgfNNet3OnlineModelWrapper* model = new AgfNNet3OnlineModelWrapper(beam, max_active, min_active, lattice_beam, acoustic_scale, frame_subsampling_factor,
        mfcc_config_filename, ie_config_filename, model_filename,
        nonterm_phones_offset, rules_nonterm_offset, dictation_nonterm_offset,
        word_syms_filename, word_align_lexicon_filename,
        top_fst_filename, dictation_fst_filename,
        verbosity);
    return model;
}

bool load_lexicon_agf_nnet3(void* model_vp, char* word_syms_filename_cp, char* word_align_lexicon_filename_cp) {
    AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    std::string word_syms_filename(word_syms_filename_cp), word_align_lexicon_filename(word_align_lexicon_filename_cp);
    bool result = model->load_lexicon(word_syms_filename, word_align_lexicon_filename);
    return result;
}

int32_t add_grammar_fst_agf_nnet3(void* model_vp, char* grammar_fst_filename_cp) {
    AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    std::string grammar_fst_filename(grammar_fst_filename_cp);
    int32_t grammar_fst_index = model->add_grammar_fst(grammar_fst_filename);
    return grammar_fst_index;
}

bool reload_grammar_fst_agf_nnet3(void* model_vp, int32_t grammar_fst_index, char* grammar_fst_filename_cp) {
    AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    std::string grammar_fst_filename(grammar_fst_filename_cp);
    bool result = model->reload_grammar_fst(grammar_fst_index, grammar_fst_filename);
    return result;
}

bool remove_grammar_fst_agf_nnet3(void* model_vp, int32_t grammar_fst_index) {
    AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    bool result = model->remove_grammar_fst(grammar_fst_index);
    return result;
}

bool decode_agf_nnet3(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize,
    bool* grammars_activity_cp, int32_t grammars_activity_cp_size, bool save_adaptation_state) {
    try {
        AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
        std::vector<bool> grammars_activity(grammars_activity_cp_size, false);
        for (size_t i = 0; i < grammars_activity_cp_size; i++) {
            grammars_activity[i] = grammars_activity_cp[i];
        }
        bool result = model->decode(samp_freq, num_frames, frames, finalize, grammars_activity, save_adaptation_state);
        return result;

    } catch(const std::exception& e) {
        KALDI_WARN << "Trying to survive fatal exception: " << e.what();
        return false;
    }
}

bool save_adaptation_state_agf_nnet3(void* model_vp) {
    try {
        AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
        bool result = model->save_adaptation_state();
        return result;

    } catch(const std::exception& e) {
        KALDI_WARN << "Trying to survive fatal exception: " << e.what();
        return false;
    }
}

bool reset_adaptation_state_agf_nnet3(void* model_vp) {
    try {
        AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
        model->reset_adaptation_state();
        return true;

    } catch(const std::exception& e) {
        KALDI_WARN << "Trying to survive fatal exception: " << e.what();
        return false;
    }
}

bool get_output_agf_nnet3(void* model_vp, char* output, int32_t output_max_length, double* likelihood_p) {
    try {
        if (output_max_length < 1) return false;
        AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
        std::string decoded_string;
        double likelihood;
        model->get_decoded_string(decoded_string, likelihood);
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

bool get_word_align_agf_nnet3(void* model_vp, int32_t* times_cp, int32_t* lengths_cp, int32_t num_words) {
    try {
        AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
        std::vector<string> words;
        std::vector<int32> times, lengths;
        bool result = model->get_word_alignment(words, times, lengths, false);

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
