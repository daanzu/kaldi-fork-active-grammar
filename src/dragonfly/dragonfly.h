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

#if defined(_MSC_VER)
    #ifdef DRAGONFLY_EXPORTS
        #define DRAGONFLY_API extern "C" __declspec(dllexport)
    #else
        #define DRAGONFLY_API extern "C" __declspec(dllimport)
    #endif
#elif defined(__GNUC__)
    // unnecessary
    #define DRAGONFLY_API extern "C" __attribute__((visibility("default")))
#else
    #define DRAGONFLY_API
    #pragma warning Unknown dynamic link import / export semantics.
#endif

#include <stdint.h>

DRAGONFLY_API void* gmm__init(float beam, int32_t max_active, int32_t min_active, float lattice_beam,
    char* word_syms_filename_cp, char* fst_in_str_cp, char* config_cp);
DRAGONFLY_API bool gmm__decode(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize);
DRAGONFLY_API bool gmm__get_output(void* model_vp, char* output, int32_t output_length, double* likelihood_p);

DRAGONFLY_API void* gmm_otf__init(float beam, int32_t max_active, int32_t min_active, float lattice_beam,
	char* word_syms_filename_cp, char* config_cp,
	char* hcl_fst_filename_cp, char** grammar_fst_filenames_cp, int32_t grammar_fst_filenames_len);
DRAGONFLY_API bool gmm_otf__add_grammar_fst(void* model_vp, char* grammar_fst_filename_cp);
DRAGONFLY_API bool gmm_otf__decode(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize,
	bool* grammars_activity, int32_t grammars_activity_size);
DRAGONFLY_API bool gmm_otf__get_output(void* model_vp, char* output, int32_t output_length, double* likelihood_p);

DRAGONFLY_API bool nnet3_base__load_lexicon(void* model_vp, char* word_syms_filename_cp, char* word_align_lexicon_filename_cp);
DRAGONFLY_API bool nnet3_base__save_adaptation_state(void* model_vp);
DRAGONFLY_API bool nnet3_base__reset_adaptation_state(void* model_vp);
DRAGONFLY_API bool nnet3_base__get_word_align(void* model_vp, int32_t* times_cp, int32_t* lengths_cp, int32_t num_words);
DRAGONFLY_API bool nnet3_base__decode(void* model_vp, float samp_freq, int32_t num_samples, float* samples, bool finalize, bool save_adaptation_state);
DRAGONFLY_API bool nnet3_base__get_output(void* model_vp, char* output, int32_t output_max_length,
        float* likelihood_p, float* am_score_p, float* lm_score_p, float* confidence_p, float* expected_error_rate_p);
DRAGONFLY_API bool nnet3_base__set_lm_prime_text(void* model_vp, char* prime_cp);

DRAGONFLY_API void* nnet3_plain__construct(char* model_dir_cp, char* config_str_cp, int32_t verbosity);
DRAGONFLY_API bool nnet3_plain__destruct(void* model_vp);
DRAGONFLY_API bool nnet3_plain__decode(void* model_vp, float samp_freq, int32_t num_samples, float* samples, bool finalize, bool save_adaptation_state);

DRAGONFLY_API void* nnet3_agf__construct(char* model_dir_cp, char* config_str_cp, int32_t verbosity);
DRAGONFLY_API bool nnet3_agf__destruct(void* model_vp);
DRAGONFLY_API int32_t nnet3_agf__add_grammar_fst(void* model_vp, void* grammar_fst_cp);
DRAGONFLY_API int32_t nnet3_agf__add_grammar_fst_file(void* model_vp, char* grammar_fst_filename_cp);
DRAGONFLY_API bool nnet3_agf__reload_grammar_fst_(void* model_vp, int32_t grammar_fst_index, void* grammar_fst_cp);
DRAGONFLY_API bool nnet3_agf__reload_grammar_fst_file(void* model_vp, int32_t grammar_fst_index, char* grammar_fst_filename_cp);
DRAGONFLY_API bool nnet3_agf__remove_grammar_fst(void* model_vp, int32_t grammar_fst_index);
DRAGONFLY_API bool nnet3_agf__decode(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize,
    bool* grammars_activity_cp, int32_t grammars_activity_cp_size, bool save_adaptation_state);
DRAGONFLY_API void* nnet3_agf__construct_compiler(char* config_str_cp);
DRAGONFLY_API bool nnet3_agf__destruct_compiler(void* compiler_vp);
DRAGONFLY_API void* nnet3_agf__compile_graph(void* compiler_vp, char* config_str_cp, void* grammar_fst_cp, bool return_graph);
DRAGONFLY_API void* nnet3_agf__compile_graph_text(void* compiler_vp, char* config_str_cp, char* grammar_fst_text_cp, bool return_graph);
DRAGONFLY_API void* nnet3_agf__compile_graph_file(void* compiler_vp, char* config_str_cp, char* grammar_fst_filename_cp, bool return_graph);

DRAGONFLY_API void* nnet3_laf__construct(char* model_dir_cp, char* config_str_cp, int32_t verbosity);
DRAGONFLY_API bool nnet3_laf__destruct(void* model_vp);
DRAGONFLY_API int32_t nnet3_laf__add_grammar_fst(void* model_vp, void* grammar_fst_cp);
DRAGONFLY_API int32_t nnet3_laf__add_grammar_fst_text(void* model_vp, char* grammar_fst_cp);
DRAGONFLY_API bool nnet3_laf__reload_grammar_fst(void* model_vp, int32_t grammar_fst_index, void* grammar_fst_cp);
DRAGONFLY_API bool nnet3_laf__remove_grammar_fst(void* model_vp, int32_t grammar_fst_index);
DRAGONFLY_API bool nnet3_laf__decode(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize,
    bool* grammars_activity_cp, int32_t grammars_activity_cp_size, bool save_adaptation_state);

DRAGONFLY_API bool fst__init(int32_t eps_like_ilabels_len, int32_t eps_like_ilabels_cp[], int32_t silent_olabels_len, int32_t silent_olabels_cp[], int32_t wildcard_olabels_len, int32_t wildcard_olabels_cp[]);
DRAGONFLY_API void* fst__construct();
DRAGONFLY_API bool fst__destruct(void* fst_vp);
DRAGONFLY_API int32_t fst__add_state(void* fst_vp, float weight, bool initial);
DRAGONFLY_API bool fst__add_arc(void* fst_vp, int32_t src_state_id, int32_t dst_state_id, int32_t ilabel, int32_t olabel, float weight);
DRAGONFLY_API bool fst__has_eps_path(void* fst_vp, int32_t path_src_state, int32_t path_dst_state);
DRAGONFLY_API bool fst__does_match(void* fst_vp, int32_t target_labels_len, int32_t target_labels_cp[], int32_t output_labels_cp[], int32_t* output_labels_len);
