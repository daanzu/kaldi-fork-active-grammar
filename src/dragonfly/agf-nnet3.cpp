// dragonfly.cpp : Defines the exported functions for the DLL application.
//

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
#include "active-grammar-fst.h"

#define VERBOSE 1

namespace dragonfly {
    using namespace kaldi;
    using namespace fst;

    ConstFst<StdArc>* CastOrConvertToConstFst(Fst<StdArc>* fst) {
        // This version currently supports ConstFst<StdArc> or VectorFst<StdArc>
        std::string real_type = fst->Type();
        KALDI_ASSERT(real_type == "vector" || real_type == "const");
        if (real_type == "const") {
            return dynamic_cast<ConstFst<StdArc>*>(fst);
        } else {
            // As the 'fst' can't cast to ConstFst, we carete a new
            // ConstFst<StdArc> initialized by 'fst', and delete 'fst'.
            ConstFst<StdArc>* new_fst = new ConstFst<StdArc>(*fst);
            delete fst;
            return new_fst;
        }
    }

    class AgfNNet3OnlineModelWrapper {
    public:

        AgfNNet3OnlineModelWrapper(BaseFloat beam, int32 max_active, int32 min_active, BaseFloat lattice_beam, BaseFloat acoustic_scale, int32 frame_subsampling_factor,
            int32 nonterm_phones_offset, std::string& word_syms_filename, std::string& mfcc_config_filename, std::string& ie_config_filename,
            std::string& model_filename, std::string& top_fst_filename);
        ~AgfNNet3OnlineModelWrapper();

        bool add_grammar_fst(std::string& grammar_fst_filename);
        bool decode(BaseFloat samp_freq, int32 num_frames, BaseFloat* frames, bool finalize, std::vector<bool>& grammars_activity);

        void get_decoded_string(std::string& decoded_string, double& likelihood);
        bool get_word_alignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths);

    protected:

        // model
        fst::SymbolTable *word_syms;
        int32 nonterm_phones_offset;
        OnlineNnet2FeaturePipelineConfig feature_config;
        nnet3::NnetSimpleLoopedComputationOptions decodable_config;
        LatticeFasterDecoderConfig decoder_config;
        OnlineEndpointConfig endpoint_config;
        OnlineNnet2FeaturePipelineInfo *feature_info;
        TransitionModel trans_model;
        nnet3::AmNnetSimple am_nnet;
        ActiveGrammarFst* active_grammar_fst;
        StdConstFst *top_fst;
        std::vector<StdFst*> grammar_fsts;
        std::map<StdFst*, std::string> grammar_fsts_name_map;
        std::vector<std::pair<int32, const StdConstFst *> > active_grammar_ifsts;
        std::vector<bool> grammar_fsts_enabled;
        std::vector<std::vector<int32> > word_alignment_lexicon;

        // decoder
        OnlineIvectorExtractorAdaptationState* adaptation_state = nullptr;
        OnlineNnet2FeaturePipeline* feature_pipeline = nullptr;
        OnlineSilenceWeighting* silence_weighting = nullptr;
        nnet3::DecodableNnetSimpleLoopedInfo* decodable_info = nullptr;
        SingleUtteranceNnet3DecoderTpl<fst::ActiveGrammarFst>* decoder = nullptr;
        std::vector<std::pair<int32, BaseFloat> > delta_weights;
        int32 tot_frames, tot_frames_decoded;
        CompactLattice best_path_clat;

        StdConstFst* read_fst_file(std::string filename);

        void start_decoding(std::vector<bool> grammars_activity);
        void free_decoder(void);
    };

    AgfNNet3OnlineModelWrapper::AgfNNet3OnlineModelWrapper(
        BaseFloat beam, int32 max_active, int32 min_active, BaseFloat lattice_beam, BaseFloat acoustic_scale, int32 frame_subsampling_factor,
        int32 nonterm_phones_offset, std::string& word_syms_filename, std::string& mfcc_config_filename, std::string& ie_config_filename,
        std::string& model_filename, std::string& top_fst_filename) {
#if VERBOSE
        KALDI_LOG << "nonterm_phones_offset: " << nonterm_phones_offset;
        KALDI_LOG << "word_syms_filename: " << word_syms_filename;
        KALDI_LOG << "mfcc_config_filename: " << mfcc_config_filename;
        KALDI_LOG << "ie_config_filename: " << ie_config_filename;
        KALDI_LOG << "model_filename: " << model_filename;
        KALDI_LOG << "top_fst_filename: " << top_fst_filename;
#else
        // silence kaldi output as well
        SetLogHandler(silent_log_handler);
#endif

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
            this->trans_model.Read(ki.Stream(), binary);
            this->am_nnet.Read(ki.Stream(), binary);
            SetBatchnormTestMode(true, &(this->am_nnet.GetNnet()));
            SetDropoutTestMode(true, &(this->am_nnet.GetNnet()));
            nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(this->am_nnet.GetNnet()));
        }

        feature_info = new OnlineNnet2FeaturePipelineInfo(feature_config);
        decodable_info = new nnet3::DecodableNnetSimpleLoopedInfo(decodable_config, &am_nnet);
        top_fst = dynamic_cast<StdConstFst*>(ReadFstKaldiGeneric(top_fst_filename));

        this->nonterm_phones_offset = nonterm_phones_offset;
        word_syms = nullptr;
        if (word_syms_filename != "")
            if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
                KALDI_ERR << "Could not read symbol table from file "
                << word_syms_filename;

        // {
        // 	bool binary_in;
        // 	Input ki(align_lex_filename, &binary_in);
        // 	KALDI_ASSERT(!binary_in && "Not expecting binary file for lexicon");
        // 	if (!ReadLexiconForWordAlign(ki.Stream(), &word_alignment_lexicon)) {
        // 		KALDI_ERR << "Error reading alignment lexicon from "
        // 			<< align_lex_filename;
        // 	}
        // }

        // NOTE: assumes single speaker
        adaptation_state = new OnlineIvectorExtractorAdaptationState(feature_info->ivector_extractor_info);

        active_grammar_fst = nullptr;
        decoder = nullptr;
        tot_frames = 0;
        tot_frames_decoded = 0;
        }

    AgfNNet3OnlineModelWrapper::~AgfNNet3OnlineModelWrapper() {
        free_decoder();
        // delete ...;
        // FIXME
    }

    StdConstFst* AgfNNet3OnlineModelWrapper::read_fst_file(std::string filename) {
        if (filename.compare(filename.length() - 4, 4, ".txt") == 0) {
            // FIXME: fstdeterminize | fstminimize | fstrmepsilon | fstarcsort --sort_type=ilabel
            KALDI_ERR << "cannot read text fst file!";
            return nullptr;
        } else {
            return dynamic_cast<StdConstFst*>(ReadFstKaldiGeneric(filename));
        }
    }

    bool AgfNNet3OnlineModelWrapper::add_grammar_fst(std::string& grammar_fst_filename) {
        auto grammar_fst = read_fst_file(grammar_fst_filename);
        auto i = grammar_fsts.size();
        KALDI_LOG << "#" << i << " 0x" << grammar_fst << " " << grammar_fst_filename;
        grammar_fsts.emplace_back(grammar_fst);
        grammar_fsts_enabled.emplace_back(false);
        grammar_fsts_name_map[grammar_fst] = grammar_fst_filename;
        active_grammar_ifsts.emplace_back(std::make_pair(nonterm_phones_offset + 4 + active_grammar_ifsts.size(), grammar_fst));
        if (active_grammar_fst) {
            delete active_grammar_fst;
            active_grammar_fst = nullptr;
        }
        return true;
    }

    void AgfNNet3OnlineModelWrapper::start_decoding(std::vector<bool> grammars_activity) {
        free_decoder();
        if (active_grammar_fst == nullptr) {
            // Timer timer(true);
            active_grammar_fst = new ActiveGrammarFst(nonterm_phones_offset, *top_fst, active_grammar_ifsts);
            // KALDI_LOG << "built new ActiveGrammarFst" << " in " << (timer.Elapsed() * 1000) << "ms.";
        }
        active_grammar_fst->UpdateActivity(grammars_activity);
        
        feature_pipeline = new OnlineNnet2FeaturePipeline(*feature_info);
        feature_pipeline->SetAdaptationState(*adaptation_state);
        silence_weighting = new OnlineSilenceWeighting(
            trans_model, feature_info->silence_weighting_config,
            decodable_config.frame_subsampling_factor);
        decoder = new SingleUtteranceNnet3DecoderTpl<fst::ActiveGrammarFst>(
            decoder_config, trans_model, *decodable_info, *active_grammar_fst, feature_pipeline);
    }

    void AgfNNet3OnlineModelWrapper::free_decoder(void) {
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
    bool AgfNNet3OnlineModelWrapper::decode(BaseFloat samp_freq, int32 num_frames, BaseFloat* frames, bool finalize,
        std::vector<bool>& grammars_activity) {
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

        if (silence_weighting->Active() && feature_pipeline->IvectorFeature() != nullptr) {
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

            feature_pipeline->GetAdaptationState(adaptation_state);

            tot_frames_decoded = tot_frames;
            tot_frames = 0;

            free_decoder();
        }

        return true;
    }

    void AgfNNet3OnlineModelWrapper::get_decoded_string(std::string& decoded_string, double& likelihood) {
        Lattice best_path_lat;

        decoded_string = "";

        if (decoder) {
            // decoding is not finished yet, so we will look up the best partial result so far

            // if (decoder->NumFramesDecoded() == 0) {
            //     likelihood = 0.0;
            //     return;
            // }

            decoder->GetBestPath(false, &best_path_lat);
        }
        else {
            ConvertLattice(best_path_clat, &best_path_lat);
        }

        std::vector<int32> words;
        std::vector<int32> alignment;
        LatticeWeight weight;
        int32 num_frames;
        GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
        num_frames = alignment.size();
        likelihood = -(weight.Value1() + weight.Value2()) / num_frames;

        for (size_t i = 0; i < words.size(); i++) {
            std::string s = word_syms->Find(words[i]);
            if (s == "")
                KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
            if (i != 0)
                decoded_string += ' ';
            decoded_string += s;
        }
    }

    bool AgfNNet3OnlineModelWrapper::get_word_alignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths) {
        return false;
    }
}

using namespace dragonfly;

void* init_agf_nnet3(float beam, int32_t max_active, int32_t min_active, float lattice_beam, float acoustic_scale, int32_t frame_subsampling_factor,
    int32_t nonterm_phones_offset, char* word_syms_filename_cp, char* mfcc_config_filename_cp, char* ie_config_filename_cp,
    char* model_filename_cp, char* top_fst_filename_cp) {
    std::string word_syms_filename(word_syms_filename_cp), mfcc_config_filename(mfcc_config_filename_cp), ie_config_filename(ie_config_filename_cp),
        model_filename(model_filename_cp), top_fst_filename(top_fst_filename_cp);
    AgfNNet3OnlineModelWrapper* model = new AgfNNet3OnlineModelWrapper(beam, max_active, min_active, lattice_beam, acoustic_scale, frame_subsampling_factor,
        nonterm_phones_offset, word_syms_filename, mfcc_config_filename, ie_config_filename, model_filename, top_fst_filename);
    return model;
}

bool add_grammar_fst_agf_nnet3(void* model_vp, char* grammar_fst_filename_cp) {
    AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    std::string grammar_fst_filename(grammar_fst_filename_cp);
    bool result = model->add_grammar_fst(grammar_fst_filename);
    return result;
}

bool decode_agf_nnet3(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize,
    bool* grammars_activity_cp, int32_t grammars_activity_cp_size) {
    AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    std::vector<bool> grammars_activity(grammars_activity_cp_size);
    for (size_t i = 0; i < grammars_activity_cp_size; i++) {
        grammars_activity[i] = grammars_activity_cp[i];
    }
    bool result = model->decode(samp_freq, num_frames, frames, finalize, grammars_activity);
    return result;
}

bool get_output_agf_nnet3(void* model_vp, char* output, int32_t output_length, double* likelihood_p) {
    if (output_length < 1) return false;
    AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    std::string decoded_string;
    double likelihood;
    model->get_decoded_string(decoded_string, likelihood);
    const char* cstr = decoded_string.c_str();
    strncpy(output, cstr, output_length);
    output[output_length - 1] = 0;
    *likelihood_p = likelihood;
    return true;
}
