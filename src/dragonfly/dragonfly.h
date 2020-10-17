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

DRAGONFLY_API void* init_gmm(float beam, int32_t max_active, int32_t min_active, float lattice_beam,
    char* word_syms_filename_cp, char* fst_in_str_cp, char* config_cp);
DRAGONFLY_API bool decode_gmm(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize);
DRAGONFLY_API bool get_output_gmm(void* model_vp, char* output, int32_t output_length, double* likelihood_p);

DRAGONFLY_API void* init_otf_gmm(float beam, int32_t max_active, int32_t min_active, float lattice_beam,
	char* word_syms_filename_cp, char* config_cp,
	char* hcl_fst_filename_cp, char** grammar_fst_filenames_cp, int32_t grammar_fst_filenames_len);
DRAGONFLY_API bool add_grammar_fst_otf_gmm(void* model_vp, char* grammar_fst_filename_cp);
DRAGONFLY_API bool decode_otf_gmm(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize,
	bool* grammars_activity, int32_t grammars_activity_size);
DRAGONFLY_API bool get_output_otf_gmm(void* model_vp, char* output, int32_t output_length, double* likelihood_p);

DRAGONFLY_API bool load_lexicon_base_nnet3(void* model_vp, char* word_syms_filename_cp, char* word_align_lexicon_filename_cp);
DRAGONFLY_API bool save_adaptation_state_base_nnet3(void* model_vp);
DRAGONFLY_API bool reset_adaptation_state_base_nnet3(void* model_vp);
DRAGONFLY_API bool get_word_align_base_nnet3(void* model_vp, int32_t* times_cp, int32_t* lengths_cp, int32_t num_words);
DRAGONFLY_API bool decode_base_nnet3(void* model_vp, float samp_freq, int32_t num_samples, float* samples, bool finalize, bool save_adaptation_state);
DRAGONFLY_API bool get_output_base_nnet3(void* model_vp, char* output, int32_t output_max_length,
        float* likelihood_p, float* am_score_p, float* lm_score_p, float* confidence_p, float* expected_error_rate_p);
DRAGONFLY_API bool set_lm_prime_text_base_nnet3(void* model_vp, char* prime_cp);

DRAGONFLY_API void* init_plain_nnet3(char* model_dir_cp, char* config_str_cp, int32_t verbosity);
DRAGONFLY_API bool decode_plain_nnet3(void* model_vp, float samp_freq, int32_t num_samples, float* samples, bool finalize, bool save_adaptation_state);

DRAGONFLY_API void* init_agf_nnet3(char* model_dir_cp, char* config_str_cp, int32_t verbosity);
DRAGONFLY_API int32_t add_grammar_fst_agf_nnet3(void* model_vp, char* grammar_fst_filename_cp);
DRAGONFLY_API bool reload_grammar_fst_agf_nnet3(void* model_vp, int32_t grammar_fst_index, char* grammar_fst_filename_cp);
DRAGONFLY_API bool remove_grammar_fst_agf_nnet3(void* model_vp, int32_t grammar_fst_index);
DRAGONFLY_API bool decode_agf_nnet3(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize,
    bool* grammars_activity_cp, int32_t grammars_activity_cp_size, bool save_adaptation_state);
DRAGONFLY_API bool compile_graph_agf(void* model_vp, int32_t argc, char** argv);

DRAGONFLY_API void* init_laf_nnet3(char* model_dir_cp, char* config_str_cp, int32_t verbosity);
DRAGONFLY_API int32_t add_grammar_fst_laf_nnet3(void* model_vp, void* grammar_fst_cp);
DRAGONFLY_API int32_t add_grammar_fst_text_laf_nnet3(void* model_vp, char* grammar_fst_cp);
DRAGONFLY_API bool reload_grammar_fst_laf_nnet3(void* model_vp, int32_t grammar_fst_index, char* grammar_fst_filename_cp);
DRAGONFLY_API bool remove_grammar_fst_laf_nnet3(void* model_vp, int32_t grammar_fst_index);
DRAGONFLY_API bool decode_laf_nnet3(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize,
    bool* grammars_activity_cp, int32_t grammars_activity_cp_size, bool save_adaptation_state);

DRAGONFLY_API bool fst__init(int32_t eps_like_ilabels_len, int32_t eps_like_ilabels_cp[], int32_t silent_olabels_len, int32_t silent_olabels_cp[], int32_t wildcard_olabels_len, int32_t wildcard_olabels_cp[]);
DRAGONFLY_API void* fst__construct();
DRAGONFLY_API bool fst__destruct(void* fst_vp);
DRAGONFLY_API int32_t fst__add_state(void* fst_vp, float weight, bool initial);
DRAGONFLY_API bool fst__add_arc(void* fst_vp, int32_t src_state_id, int32_t dst_state_id, int32_t ilabel, int32_t olabel, float weight);
DRAGONFLY_API bool fst__has_eps_path(void* fst_vp, int32_t path_src_state, int32_t path_dst_state);
DRAGONFLY_API bool fst__does_match(void* fst_vp, int32_t target_labels_len, int32_t target_labels_cp[], int32_t output_labels_cp[], int32_t* output_labels_len);
