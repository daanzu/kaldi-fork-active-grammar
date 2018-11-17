// dragonfly.cpp : Defines the exported functions for the DLL application.
//

extern "C" {
#include "dragonfly.h"
}

#include "feat/wave-reader.h"
#include "online2/online-feature-pipeline.h"
#include "online2/online-gmm-decoding.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "lat/word-align-lattice-lexicon.h"

#define VERBOSE 1

namespace dragonfly
{
	using namespace kaldi;

	class GmmOnlineModelWrapper
	{
	public:

		GmmOnlineModelWrapper(BaseFloat beam, int32 max_active, int32 min_active, BaseFloat lattice_beam,
			std::string & word_syms_filename, std::string & fst_in_str, std::string & config);
		~GmmOnlineModelWrapper();

		bool decode(BaseFloat samp_freq, int32 num_frames, BaseFloat * frames, bool finalize);

		void get_decoded_string(std::string & decoded_string, double & likelihood);
		bool get_word_alignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths);

	private:

		// model
		fst::SymbolTable *word_syms;
		OnlineGmmDecodingConfig decode_config;
		OnlineFeaturePipelineCommandLineConfig feature_cmdline_config;
		OnlineFeaturePipelineConfig *feature_config;
		OnlineFeaturePipeline *feature_pipeline_prototype;
		OnlineEndpointConfig endpoint_config;
		OnlineGmmDecodingModels *gmm_models;
		fst::Fst<fst::StdArc> *decode_fst;
		std::vector<std::vector<int32> > word_alignment_lexicon;

		// decoder
		OnlineGmmAdaptationState *adaptation_state;
		SingleUtteranceGmmDecoder *decoder;
		int32 tot_frames, tot_frames_decoded;
		CompactLattice best_path_clat;

		void start_decoding(void);
		void free_decoder(void);
	};

	// struct GmmModel
	// {
	//     // model
	//     fst::SymbolTable *word_syms;
	//     OnlineGmmDecodingConfig decode_config;
	//     OnlineFeaturePipelineCommandLineConfig feature_cmdline_config;
	//     OnlineFeaturePipelineConfig *feature_config;
	//     OnlineFeaturePipeline *feature_pipeline_prototype;
	//     OnlineEndpointConfig endpoint_config;
	//     OnlineGmmDecodingModels *gmm_models;
	//     fst::Fst<fst::StdArc> *decode_fst;
	//     std::vector<std::vector<int32> > word_alignment_lexicon;
	//     // decoder
	//     OnlineGmmAdaptationState *adaptation_state;
	//     SingleUtteranceGmmDecoder *decoder;
	//     int32 tot_frames, tot_frames_decoded;
	//     CompactLattice best_path_clat;
	// };

	void silent_log_handler(const LogMessageEnvelope &envelope,
		const char *message) {
		// nothing - this handler simply keeps silent
	}

	GmmOnlineModelWrapper::GmmOnlineModelWrapper(BaseFloat beam, int32 max_active, int32 min_active, BaseFloat lattice_beam,
		std::string & word_syms_filename, std::string & fst_in_str, std::string & config)
	{
#if VERBOSE
		KALDI_LOG << "config: " << config;
#else
		// silence kaldi output as well
		SetLogHandler(silent_log_handler);
#endif

		ParseOptions po("");
		feature_cmdline_config.Register(&po);
		decode_config.Register(&po);
		endpoint_config.Register(&po);
		po.ReadConfigFile(config);

		decode_config.faster_decoder_opts.max_active = max_active;
		decode_config.faster_decoder_opts.min_active = min_active;
		decode_config.faster_decoder_opts.beam = beam;
		decode_config.faster_decoder_opts.lattice_beam = lattice_beam;

		feature_config = new OnlineFeaturePipelineConfig(feature_cmdline_config);
		feature_pipeline_prototype = new OnlineFeaturePipeline(*this->feature_config);

		gmm_models = new OnlineGmmDecodingModels(decode_config);
		decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);

		word_syms = NULL;
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

		decoder = NULL;
		adaptation_state = NULL;
		tot_frames = 0;
		tot_frames_decoded = 0;
	}

	GmmOnlineModelWrapper::~GmmOnlineModelWrapper()
	{
		free_decoder();
		delete feature_config;
		delete feature_pipeline_prototype;
		delete gmm_models;
	}

	void GmmOnlineModelWrapper::start_decoding(void)
	{
		free_decoder();
		adaptation_state = new OnlineGmmAdaptationState();
		decoder = new SingleUtteranceGmmDecoder(decode_config,
			*gmm_models,
			*feature_pipeline_prototype,
			*decode_fst,
			*adaptation_state);
	}

	void GmmOnlineModelWrapper::free_decoder(void)
	{
		if (decoder) {
			delete decoder;
			decoder = NULL;
		}
		if (adaptation_state) {
			delete adaptation_state;
			adaptation_state = NULL;
		}
	}

	bool GmmOnlineModelWrapper::decode(BaseFloat samp_freq, int32 num_frames, BaseFloat * frames, bool finalize)
	{
		using fst::VectorFst;

		if (!decoder)
			start_decoding();

		Vector<BaseFloat> wave_part(num_frames, kUndefined);
		for (int i = 0; i<num_frames; i++) {
			wave_part(i) = frames[i];
		}
		tot_frames += num_frames;

		decoder->FeaturePipeline().AcceptWaveform(samp_freq, wave_part);

		if (finalize) {
			// no more input; flush out last frames
			decoder->FeaturePipeline().InputFinished();
		}

		decoder->AdvanceDecoding();

		if (finalize) {
			decoder->FinalizeDecoding();

			CompactLattice clat;
			bool end_of_utterance = true;
			decoder->EstimateFmllr(end_of_utterance);
			bool rescore_if_needed = true;
			decoder->GetLattice(rescore_if_needed, end_of_utterance, &clat);

			if (clat.NumStates() == 0) {
				KALDI_WARN << "Empty lattice.";
				return false;
			}

			CompactLatticeShortestPath(clat, &best_path_clat);

			tot_frames_decoded = tot_frames;
			tot_frames = 0;

			free_decoder();
		}

		return true;
	}

	void GmmOnlineModelWrapper::get_decoded_string(std::string & decoded_string, double & likelihood)
	{
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
			decoded_string += s + ' ';
		}
	}

	bool GmmOnlineModelWrapper::get_word_alignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths)
	{
		return false;
	}
}

int test()
{
	return 42;
}

using namespace dragonfly;

void* init_gmm(float beam, int32_t max_active, int32_t min_active, float lattice_beam,
	char* word_syms_filename_cp, char* fst_in_str_cp, char* config_cp)
{
	std::string word_syms_filename(word_syms_filename_cp), fst_in_str(fst_in_str_cp), config(config_cp);
	GmmOnlineModelWrapper* model = new GmmOnlineModelWrapper(beam, max_active, min_active, lattice_beam,
		word_syms_filename, fst_in_str, config);
	return model;
}

bool decode_gmm(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize)
{
	GmmOnlineModelWrapper* model = static_cast<GmmOnlineModelWrapper*>(model_vp);
	bool result = model->decode(samp_freq, num_frames, frames, finalize);
	return result;
}

void get_output_gmm(void* model_vp, char* output, int32_t output_length, double* likelihood_p)
{
	GmmOnlineModelWrapper* model = static_cast<GmmOnlineModelWrapper*>(model_vp);
	std::string decoded_string;
	double likelihood;
	model->get_decoded_string(decoded_string, likelihood);
	const char* cstr = decoded_string.c_str();
	strncpy(output, cstr, output_length);
	*likelihood_p = likelihood;
}
