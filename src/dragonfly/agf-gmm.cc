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
#include "online2/online-gmm-decoding.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "lat/word-align-lattice-lexicon.h"
#include "active-grammar-fst.h"

#define VERBOSE 1

namespace dragonfly
{
	using namespace kaldi;
	using namespace fst;

	CacheOptions agf_default_cache_opts(false, 128 * 1024 * 1024);  // default: true, 1 * 1024 * 1024

	ComposeFst<StdArc>* OTFComposeFst(const StdFst &ifst1, const StdFst &ifst2, const CacheOptions& cache_opts = agf_default_cache_opts);
	ComposeFst<StdArc>* OTFLaComposeFst(const StdFst &ifst1, const StdFst &ifst2, const CacheOptions& cache_opts = agf_default_cache_opts);

	// ComposeFst<StdArc>* OTFComposeFst(const StdFst &ifst1, const StdFst &ifst2, const CacheOptions& cache_opts = default_cache_opts)
	// {
	// 	return new ComposeFst<StdArc>(ifst1, ifst2, cache_opts);
	// }

	// ComposeFst<StdArc>* OTFLaComposeFst(const StdFst &ifst1, const StdFst &ifst2, const CacheOptions& cache_opts = default_cache_opts)
	// {
	// 	typedef LookAheadMatcher<StdFst> M;
	// 	typedef AltSequenceComposeFilter<M> SF;
	// 	typedef LookAheadComposeFilter<SF, M>  LF;
	// 	typedef PushWeightsComposeFilter<LF, M> WF;
	// 	typedef PushLabelsComposeFilter<WF, M> ComposeFilter;
	// 	typedef M FstMatcher;
	// 	ComposeFstOptions<StdArc, FstMatcher, ComposeFilter> opts(cache_opts);
	// 	return new ComposeFst<StdArc>(ifst1, ifst2, opts);
	// }

	class AgfGmmOnlineModelWrapper
	{
	public:

		AgfGmmOnlineModelWrapper(BaseFloat beam, int32 max_active, int32 min_active, BaseFloat lattice_beam,
			std::string& word_syms_filename, std::string& config,
			std::string& hcl_fst_filename, std::vector<std::string>& grammar_fst_filenames);
		~AgfGmmOnlineModelWrapper();

		bool add_grammar_fst(std::string& grammar_fst_filename);
		bool decode(BaseFloat samp_freq, int32 num_frames, BaseFloat* frames, bool finalize, std::vector<bool>& grammars_activity);

		void get_decoded_string(std::string& decoded_string, double& likelihood);
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
		StdFst *decode_fst = NULL;
		StdFst *hcl_fst;
		StdVectorFst *null_fst;
		size_t grammar_fsts_filled;
		std::vector<StdFst*> grammar_fsts;  // invariant: size is power of 2; grammar_fsts_enabled.size() == grammar_fsts.size()
		std::map<StdFst*, std::string> grammar_fsts_name_map;
		std::vector<bool> grammar_fsts_enabled;  // invariant: size is power of 2; grammar_fsts_enabled.size() == grammar_fsts.size()
		std::vector<UnionFst<StdArc>* > union_fsts;
		ActiveGrammarFst* active_grammar_fst;
		std::vector<std::vector<int32> > word_alignment_lexicon;

		// decoder
		OnlineGmmAdaptationState *adaptation_state;
		SingleUtteranceGmmDecoder *decoder;
		int32 tot_frames, tot_frames_decoded;
		CompactLattice best_path_clat;

		StdFst* read_fst_file(std::string filename);
		void resize_grammar_fsts(size_t target);
		StdFst* unionize_fsts(StdFst* left_fst, StdFst* right_fst);
		StdFst* unionize_fsts(const std::vector<StdFst*>& fsts, std::vector<UnionFst<StdArc>*>& union_fsts_alloced);
		size_t index_union_fst(size_t index, size_t level);
		void build_union_pyramid(const std::vector<bool>& grammars_activity);
		bool rebuild_union_pyramid(const std::vector<bool>& grammars_activity, bool force = false, size_t index = 0, size_t level = 0);

		void start_decoding(std::vector<bool> grammars_activity);
		void free_decoder(void);
	};

	AgfGmmOnlineModelWrapper::AgfGmmOnlineModelWrapper(BaseFloat beam, int32 max_active, int32 min_active, BaseFloat lattice_beam,
		std::string& word_syms_filename, std::string& config,
		std::string& hcl_fst_filename, std::vector<std::string>& grammar_fst_filenames)
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

		grammar_fsts_filled = 0;
		resize_grammar_fsts(2);
		for (size_t i = 0; i < grammar_fst_filenames.size(); i++) {
			add_grammar_fst(grammar_fst_filenames[i]);
		}

		null_fst = new StdVectorFst();
		// FIXME: make a ConstFst from this
		null_fst->AddState();
		null_fst->SetStart(0);
		null_fst->SetFinal(0, 0);
		//null_fst.AddArc(0, StdArc(134433, 0, 0, 0));
		decode_fst = OTFLaComposeFst(*hcl_fst, *null_fst);

		// build decode graph
		if (0 && grammar_fsts.size() == 1) {
			decode_fst = OTFLaComposeFst(*hcl_fst, *grammar_fsts[0]);
		} else if (0 && grammar_fsts.size() == 2) {
			auto union_fst = new UnionFst<StdArc>(*grammar_fsts[0], *grammar_fsts[1]);
			decode_fst = OTFLaComposeFst(*hcl_fst, *union_fst);
		} else if (0 && grammar_fsts.size() == 4) {
			auto union_fst1 = new UnionFst<StdArc>(*grammar_fsts[0], *grammar_fsts[1]);
			auto union_fst2 = new UnionFst<StdArc>(*grammar_fsts[2], *grammar_fsts[3]);
			//union_fst2 = new UnionFst<StdArc>(*grammar_fsts[2], null_fst);
			auto union_fst = new UnionFst<StdArc>(*union_fst1, *union_fst2);
			decode_fst = OTFLaComposeFst(*hcl_fst, *union_fst);
			grammar_fsts_enabled.flip();
			KALDI_LOG << "4 grammar_fsts";
		} else {
			//rebuild_union_pyramid(grammar_fsts_enabled, true);
			build_union_pyramid(grammar_fsts_enabled);
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

	AgfGmmOnlineModelWrapper::~AgfGmmOnlineModelWrapper()
	{
		free_decoder();
		delete feature_config;
		delete feature_pipeline_prototype;
		delete gmm_models;
		// FIXME
	}

	StdFst* AgfGmmOnlineModelWrapper::read_fst_file(std::string filename)
	{
		if (filename.compare(filename.length() - 4, 4, ".txt") == 0) {
			// FIXME: fstdeterminize | fstminimize | fstrmepsilon | fstarcsort --sort_type=ilabel
			KALDI_ERR << "cannot read text fst file!";
			return nullptr;
		} else {
			return fst::ReadFstKaldiGeneric(filename);
		}
	}

	void AgfGmmOnlineModelWrapper::resize_grammar_fsts(size_t target)
	{
		if (target > grammar_fsts.size()) {
			// ensure grammar_fsts and grammar_fsts_enabled are always a power of 2
			target = std::pow(2, std::ceil(std::log2(target)));
			grammar_fsts.resize(target, null_fst);
			grammar_fsts_enabled.resize(target, false);
		}
	}

	bool AgfGmmOnlineModelWrapper::add_grammar_fst(std::string& grammar_fst_filename)
	{
		auto grammar_fst = read_fst_file(grammar_fst_filename);
		auto i = grammar_fsts_filled;
		KALDI_LOG << "#" << i << " 0x" << grammar_fst << " " << grammar_fst_filename;
		resize_grammar_fsts(i + 1);
		grammar_fsts[i] = grammar_fst;
		grammar_fsts_enabled[i] = false;
		grammar_fsts_filled += 1;
		grammar_fsts_name_map[grammar_fst] = grammar_fst_filename;
		if (decode_fst) {
			delete decode_fst;
			decode_fst = NULL;
		}
		return true;
	}

	// intelligently return an online fst that is the union of 2 given fsts, each of which could be null_fst
	inline StdFst* AgfGmmOnlineModelWrapper::unionize_fsts(StdFst* left_fst, StdFst* right_fst)
	{
		if (left_fst != null_fst && right_fst != null_fst)
			return new UnionFst<StdArc>(*left_fst, *right_fst);
		if (left_fst != null_fst)
			return left_fst;
		if (right_fst != null_fst)
			return right_fst;
		return null_fst;
	}

	// intelligently return an online fst that is the union of all given fsts
	StdFst* AgfGmmOnlineModelWrapper::unionize_fsts(const std::vector<StdFst*>& fsts, std::vector<UnionFst<StdArc>*>& union_fsts_alloced)
	{
		if (fsts.size() <= 0) {
			KALDI_LOG << "empty fsts vector";
			return null_fst;
		} else if (fsts.size() == 1) {
			KALDI_LOG << "using grammar_fst " << grammar_fsts_name_map[fsts[0]];
			return fsts[0];
		} else if (fsts.size() == 2) {
			KALDI_LOG << "using grammar_fst " << grammar_fsts_name_map[fsts[0]];
			KALDI_LOG << "using grammar_fst " << grammar_fsts_name_map[fsts[1]];
			auto fst = new UnionFst<StdArc>(*fsts[0], *fsts[1]);
			union_fsts_alloced.emplace_back(fst);
			return fst;
		} else {
			size_t const half_size = fsts.size() / 2;
			std::vector<StdFst*> split_lo(fsts.begin(), fsts.begin() + half_size);
			std::vector<StdFst*> split_hi(fsts.begin() + half_size, fsts.end());
			auto fst = new UnionFst<StdArc>(*unionize_fsts(split_lo, union_fsts_alloced), *unionize_fsts(split_hi, union_fsts_alloced));
			union_fsts_alloced.emplace_back(fst);
			return fst;
		}
	}

	inline size_t AgfGmmOnlineModelWrapper::index_union_fst(size_t index, size_t level)
	{
		size_t offset = std::pow(2, level) - 1;
		return index + offset;
	}

	void AgfGmmOnlineModelWrapper::build_union_pyramid(const std::vector<bool>& grammars_activity)
	{
		//std::vector<StdFst*> fsts(0);
		//for (size_t i = 0; i < grammar_fsts.size(); i++) {
		//	if (grammars_activity[i]) fsts.emplace_back(grammar_fsts[i]);
		//}
		//for (auto f : union_fsts) delete f;
		//union_fsts.resize(0);
		//auto fst = unionize_fsts(fsts, union_fsts);
		//KALDI_LOG << union_fsts.size() << " union_fsts alloced";
		if (decode_fst)
			delete decode_fst;
		//std::vector<const Fst<StdArc> *> fsts(0);
		//std::vector<const Fst<StdArc> *> fsts(grammar_fsts);
		std::vector<const Fst<StdArc> *> fsts(grammar_fsts.begin(), grammar_fsts.end());
		auto fst = ActiveGrammarFst(fsts);
		decode_fst = OTFLaComposeFst(*hcl_fst, fst);
		grammar_fsts_enabled = grammars_activity;
	}

	bool AgfGmmOnlineModelWrapper::rebuild_union_pyramid(const std::vector<bool>& grammars_activity,
		bool force /* = false */, size_t index /* = 0 */, size_t level /* = 0 */)
	{
		if (level == 0) {
			// initialize; at root, index=0, level=0
			AssertEqual(grammars_activity.size(), grammar_fsts_enabled.size());
			AssertEqual(grammar_fsts.size(), grammar_fsts_enabled.size());
			AssertEqual(union_fsts.size(), grammar_fsts.size() - 1);
		}

		// at this level and index, we set union_fsts[index_union_fst(index, level)]
		size_t i = index * 2;  // index of next level up: either recursing or hitting leaves

		if (index_union_fst(0, level+1) < union_fsts.size()) {
			// recurse, rebuilding
			bool rebuilt_left = rebuild_union_pyramid(grammars_activity, force, i, level + 1);
			bool rebuilt_right = rebuild_union_pyramid(grammars_activity, force, i + 1, level + 1);
			if (rebuilt_left || rebuilt_right) {
				auto left_fst = union_fsts[index_union_fst(i, level + 1)];
				auto right_fst = union_fsts[index_union_fst(i + 1, level + 1)];
				//if (union_fsts[index_union_fst(index, level)] != null_fst) delete union_fsts[index_union_fst(index, level)];
				//union_fsts[index_union_fst(index, level)] = unionize_fsts(left_fst, right_fst);
			} else return false;

		} else {
			// rebuild leaf
			if (force || (grammar_fsts_enabled[i] != grammars_activity[i]) || (grammar_fsts_enabled[i+1] != grammars_activity[i+1])) {
				auto left_fst = (grammars_activity[i]) ? grammar_fsts[i] : null_fst;
				auto right_fst = (grammars_activity[i+1]) ? grammar_fsts[i+1] : null_fst;
				//union_fsts[index_union_fst(index, level)] = unionize_fsts(left_fst, right_fst);
				grammar_fsts_enabled[i] = grammars_activity[i];
				grammar_fsts_enabled[i+1] = grammars_activity[i+1];
			} else return false;
		}

		if (level == 0) {
			// above must not have returned false
			decode_fst = OTFLaComposeFst(*hcl_fst, *union_fsts.front());
		}
		return true;
	}

	void AgfGmmOnlineModelWrapper::start_decoding(std::vector<bool> grammars_activity)
	{
		free_decoder();
		adaptation_state = new OnlineGmmAdaptationState();
		grammars_activity.resize(grammar_fsts_enabled.size(), false);
		if (grammar_fsts_enabled != grammars_activity || false) {
			Timer timer(true);
			//rebuild_union_pyramid(grammars_activity);
			build_union_pyramid(grammars_activity);
			KALDI_LOG << "rebuilt union pyramid" << " in " << (timer.Elapsed() * 1000) << "ms.";
		}
		decoder = new SingleUtteranceGmmDecoder(decode_config,
			*gmm_models,
			*feature_pipeline_prototype,
			*decode_fst,
			*adaptation_state);
	}

	void AgfGmmOnlineModelWrapper::free_decoder(void)
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

	// grammars_activity is ignored once decoding has already started
	bool AgfGmmOnlineModelWrapper::decode(BaseFloat samp_freq, int32 num_frames, BaseFloat* frames, bool finalize,
		std::vector<bool>& grammars_activity)
	{
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

	void AgfGmmOnlineModelWrapper::get_decoded_string(std::string& decoded_string, double& likelihood)
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

	bool AgfGmmOnlineModelWrapper::get_word_alignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths)
	{
		return false;
	}
}

using namespace dragonfly;

void* init_agf_gmm(float beam, int32_t max_active, int32_t min_active, float lattice_beam,
	char* word_syms_filename_cp, char* config_cp,
	char* hcl_fst_filename_cp, char** grammar_fst_filenames_cp, int32_t grammar_fst_filenames_size)
{
	std::string word_syms_filename(word_syms_filename_cp), config(config_cp), hcl_fst_filename(hcl_fst_filename_cp);
	std::vector<std::string> grammar_fst_filenames(grammar_fst_filenames_size);
	for (size_t i = 0; i < grammar_fst_filenames_size; i++)
	{
		grammar_fst_filenames[i] = grammar_fst_filenames_cp[i];
	}
	AgfGmmOnlineModelWrapper* model = new AgfGmmOnlineModelWrapper(beam, max_active, min_active, lattice_beam,
		word_syms_filename, config, hcl_fst_filename, grammar_fst_filenames);
	return model;
}

bool add_grammar_fst_agf_gmm(void* model_vp, char* grammar_fst_filename_cp)
{
	AgfGmmOnlineModelWrapper* model = static_cast<AgfGmmOnlineModelWrapper*>(model_vp);
	std::string grammar_fst_filename(grammar_fst_filename_cp);
	bool result = model->add_grammar_fst(grammar_fst_filename);
	return result;
}

bool decode_agf_gmm(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize,
	bool* grammars_activity_cp, int32_t grammars_activity_cp_size)
{
	AgfGmmOnlineModelWrapper* model = static_cast<AgfGmmOnlineModelWrapper*>(model_vp);
	std::vector<bool> grammars_activity(grammars_activity_cp_size);
	for (size_t i = 0; i < grammars_activity_cp_size; i++)
	{
		grammars_activity[i] = grammars_activity_cp[i];
	}
	bool result = model->decode(samp_freq, num_frames, frames, finalize, grammars_activity);
	return result;
}

bool get_output_agf_gmm(void* model_vp, char* output, int32_t output_length, double* likelihood_p)
{
	if (output_length < 1) return false;
	AgfGmmOnlineModelWrapper* model = static_cast<AgfGmmOnlineModelWrapper*>(model_vp);
	std::string decoded_string;
	double likelihood;
	model->get_decoded_string(decoded_string, likelihood);
	const char* cstr = decoded_string.c_str();
	strncpy(output, cstr, output_length);
	output[output_length - 1] = 0;
	*likelihood_p = likelihood;
	return true;
}
