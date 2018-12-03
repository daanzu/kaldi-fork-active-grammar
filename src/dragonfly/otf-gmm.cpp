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
	using namespace fst;

	class OtfGmmOnlineModelWrapper
	{
	public:

		OtfGmmOnlineModelWrapper(BaseFloat beam, int32 max_active, int32 min_active, BaseFloat lattice_beam,
			std::string & word_syms_filename, std::string & config,
			std::string & hcl_fst_filename, std::vector<std::string> & grammar_fst_filenames);
		~OtfGmmOnlineModelWrapper();

		bool decode(BaseFloat samp_freq, int32 num_frames, BaseFloat * frames, bool finalize, std::vector<bool> grammars_activity);

		void get_decoded_string(std::string & decoded_string, double & likelihood);
		bool get_word_alignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths);

	protected:

		// model
		fst::SymbolTable *word_syms;
		OnlineGmmDecodingConfig decode_config;
		OnlineFeaturePipelineCommandLineConfig feature_cmdline_config;
		OnlineFeaturePipelineConfig *feature_config;
		OnlineFeaturePipeline *feature_pipeline_prototype;
		OnlineEndpointConfig endpoint_config;
		OnlineGmmDecodingModels *gmm_models;
		fst::Fst<fst::StdArc> *decode_fst;
		fst::Fst<fst::StdArc> *hcl_fst;
		std::vector<fst::Fst<fst::StdArc>* > grammar_fsts;
		std::vector<std::vector<int32> > word_alignment_lexicon;

		// decoder
		OnlineGmmAdaptationState *adaptation_state;
		SingleUtteranceGmmDecoder *decoder;
		int32 tot_frames, tot_frames_decoded;
		CompactLattice best_path_clat;

		void start_decoding(void);
		void free_decoder(void);
	};

	ComposeFst<StdArc>* OTFComposeFst(const StdFst &ifst1, const StdFst &ifst2, const CacheOptions& cache_opts = CacheOptions()) {
		return new ComposeFst<StdArc>(ifst1, ifst2, cache_opts);
	}

	ComposeFst<StdArc>* OTFLaComposeFst(const StdFst &ifst1, const StdFst &ifst2, const CacheOptions& cache_opts = CacheOptions()) {
		typedef LookAheadMatcher<StdFst> M;
		typedef AltSequenceComposeFilter<M> SF;
		typedef LookAheadComposeFilter<SF, M>  LF;
		typedef PushWeightsComposeFilter<LF, M> WF;
		typedef PushLabelsComposeFilter<WF, M> ComposeFilter;
		typedef M FstMatcher;
		ComposeFstOptions<StdArc, FstMatcher, ComposeFilter> opts(cache_opts);
		return new ComposeFst<StdArc>(ifst1, ifst2, opts);
	}

	OtfGmmOnlineModelWrapper::OtfGmmOnlineModelWrapper(BaseFloat beam, int32 max_active, int32 min_active, BaseFloat lattice_beam,
		std::string & word_syms_filename, std::string & config,
		std::string & hcl_fst_filename, std::vector<std::string> & grammar_fst_filenames)
	{
#if VERBOSE
		KALDI_LOG << "word_syms_filename: " << word_syms_filename;
		KALDI_LOG << "config: " << config;
		KALDI_LOG << "hcl_fst_filename: " << hcl_fst_filename;
		//KALDI_LOG << "grammar_fst_filenames: " << grammar_fst_filenames;
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
		hcl_fst = fst::ReadFstKaldiGeneric(hcl_fst_filename);

		grammar_fsts.resize(grammar_fst_filenames.size());
		for (size_t i = 0; i < grammar_fst_filenames.size(); i++)
		{
			auto filename = grammar_fst_filenames[i];
			if (filename.compare(filename.length() - 4, 4, ".txt") == 0)
			{
				// fstdeterminize | fstminimize | fstrmepsilon | fstarcsort --sort_type=ilabel
			}
			else {
				grammar_fsts[i] = fst::ReadFstKaldiGeneric(filename);
			}
		}

		if (grammar_fst_filenames.size() == 1)
		{
			decode_fst = OTFLaComposeFst(*hcl_fst, *grammar_fsts[0]);
		}
		else if (grammar_fst_filenames.size() == 2) {
			decode_fst = OTFLaComposeFst(*hcl_fst, UnionFst<StdArc>(*grammar_fsts[0], *grammar_fsts[1]));
			//decode_fst = UnionFst<StdArc>(OTFLaComposeFst(*hcl_fst, *grammar_fsts[0]), OTFLaComposeFst(*hcl_fst, *grammar_fsts[1]));
		}

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

	OtfGmmOnlineModelWrapper::~OtfGmmOnlineModelWrapper()
	{
		free_decoder();
		delete feature_config;
		delete feature_pipeline_prototype;
		delete gmm_models;
		// FIXME
	}

	void OtfGmmOnlineModelWrapper::start_decoding(void)
	{
		free_decoder();
		adaptation_state = new OnlineGmmAdaptationState();
		decoder = new SingleUtteranceGmmDecoder(decode_config,
			*gmm_models,
			*feature_pipeline_prototype,
			*decode_fst,
			*adaptation_state);
	}

	void OtfGmmOnlineModelWrapper::free_decoder(void)
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

	bool OtfGmmOnlineModelWrapper::decode(BaseFloat samp_freq, int32 num_frames, BaseFloat * frames, bool finalize,
		std::vector<bool> grammars_activity)
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

	void OtfGmmOnlineModelWrapper::get_decoded_string(std::string & decoded_string, double & likelihood)
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
			if (i != 0)
				decoded_string += ' ';
			decoded_string += s;
		}
	}

	bool OtfGmmOnlineModelWrapper::get_word_alignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths)
	{
		return false;
	}
}

using namespace dragonfly;

void* init_otf_gmm(float beam, int32_t max_active, int32_t min_active, float lattice_beam,
	char* word_syms_filename_cp, char* config_cp,
	char* hcl_fst_filename_cp, char** grammar_fst_filenames_cp, int32_t grammar_fst_filenames_size)
{
	std::string word_syms_filename(word_syms_filename_cp), config(config_cp), hcl_fst_filename(hcl_fst_filename_cp);
	std::vector<std::string> grammar_fst_filenames(grammar_fst_filenames_size);
	for (size_t i = 0; i < grammar_fst_filenames_size; i++)
	{
		grammar_fst_filenames[i] = grammar_fst_filenames_cp[i];
	}
	OtfGmmOnlineModelWrapper* model = new OtfGmmOnlineModelWrapper(beam, max_active, min_active, lattice_beam,
		word_syms_filename, config, hcl_fst_filename, grammar_fst_filenames);
	return model;
}

bool decode_otf_gmm(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize,
	bool* grammars_activity_cp, int32_t grammars_activity_cp_size)
{
	OtfGmmOnlineModelWrapper* model = static_cast<OtfGmmOnlineModelWrapper*>(model_vp);
	std::vector<bool> grammars_activity(grammars_activity_cp_size);
	for (size_t i = 0; i < grammars_activity_cp_size; i++)
	{
		grammars_activity[i] = grammars_activity_cp[i];
	}
	bool result = model->decode(samp_freq, num_frames, frames, finalize, grammars_activity);
	return result;
}

bool get_output_otf_gmm(void* model_vp, char* output, int32_t output_length, double* likelihood_p)
{
	if (output_length < 1) return false;
	OtfGmmOnlineModelWrapper* model = static_cast<OtfGmmOnlineModelWrapper*>(model_vp);
	std::string decoded_string;
	double likelihood;
	model->get_decoded_string(decoded_string, likelihood);
	const char* cstr = decoded_string.c_str();
	strncpy(output, cstr, output_length);
	output[output_length - 1] = 0;
	*likelihood_p = likelihood;
	return true;
}
