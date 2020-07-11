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
        #define DRAGONFLY_API __declspec(dllexport)
    #else
        #define DRAGONFLY_API __declspec(dllimport)
    #endif
#elif defined(__GNUC__)
    // unnecessary
    #define DRAGONFLY_API __attribute__((visibility("default")))
#else
    #define DRAGONFLY_API
    #pragma warning Unknown dynamic link import / export semantics.
#endif

#include <stdint.h>

extern "C" DRAGONFLY_API void* init_gmm(float beam, int32_t max_active, int32_t min_active, float lattice_beam,
    char* word_syms_filename_cp, char* fst_in_str_cp, char* config_cp);
extern "C" DRAGONFLY_API bool decode_gmm(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize);
extern "C" DRAGONFLY_API bool get_output_gmm(void* model_vp, char* output, int32_t output_length, double* likelihood_p);

extern "C" DRAGONFLY_API void* init_otf_gmm(float beam, int32_t max_active, int32_t min_active, float lattice_beam,
	char* word_syms_filename_cp, char* config_cp,
	char* hcl_fst_filename_cp, char** grammar_fst_filenames_cp, int32_t grammar_fst_filenames_len);
extern "C" DRAGONFLY_API bool add_grammar_fst_otf_gmm(void* model_vp, char* grammar_fst_filename_cp);
extern "C" DRAGONFLY_API bool decode_otf_gmm(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize,
	bool* grammars_activity, int32_t grammars_activity_size);
extern "C" DRAGONFLY_API bool get_output_otf_gmm(void* model_vp, char* output, int32_t output_length, double* likelihood_p);

bool load_lexicon_base_nnet3(void* model_vp, char* word_syms_filename_cp, char* word_align_lexicon_filename_cp);
bool save_adaptation_state_base_nnet3(void* model_vp);
bool reset_adaptation_state_base_nnet3(void* model_vp);
bool get_word_align_base_nnet3(void* model_vp, int32_t* times_cp, int32_t* lengths_cp, int32_t num_words);
bool decode_base_nnet3(void* model_vp, float samp_freq, int32_t num_samples, float* samples, bool finalize, bool save_adaptation_state);
bool get_output_base_nnet3(void* model_vp, char* output, int32_t output_max_length,
        float* likelihood_p, float* am_score_p, float* lm_score_p, float* confidence_p, float* expected_error_rate_p);

extern "C" DRAGONFLY_API void* init_plain_nnet3(float beam, int32_t max_active, int32_t min_active, float lattice_beam, float acoustic_scale, int32_t frame_subsampling_factor,
    char* model_dir_cp, char* mfcc_config_filename_cp, char* ie_config_filename_cp, char* model_filename_cp,
    char* word_syms_filename_cp, char* word_align_lexicon_filename_cp, char* fst_filename_cp,
    int32_t verbosity);
extern "C" DRAGONFLY_API bool save_adaptation_state_plain_nnet3(void* model_vp);
extern "C" DRAGONFLY_API bool reset_adaptation_state_plain_nnet3(void* model_vp);
extern "C" DRAGONFLY_API bool get_word_align_plain_nnet3(void* model_vp, int32_t* times_cp, int32_t* lengths_cp, int32_t num_words);
extern "C" DRAGONFLY_API bool decode_plain_nnet3(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize, bool save_adaptation_state);
extern "C" DRAGONFLY_API bool get_output_plain_nnet3(void* model_vp, char* output, int32_t output_max_length, double* likelihood_p);

extern "C" DRAGONFLY_API void* init_agf_nnet3(char* model_dir_cp, char* config_str_cp, int32_t verbosity);
extern "C" DRAGONFLY_API bool load_lexicon_agf_nnet3(void* model_vp, char* word_syms_filename_cp, char* word_align_lexicon_filename_cp);
extern "C" DRAGONFLY_API bool save_adaptation_state_agf_nnet3(void* model_vp);
extern "C" DRAGONFLY_API bool reset_adaptation_state_agf_nnet3(void* model_vp);
extern "C" DRAGONFLY_API bool get_word_align_agf_nnet3(void* model_vp, int32_t* times_cp, int32_t* lengths_cp, int32_t num_words);
extern "C" DRAGONFLY_API int32_t add_grammar_fst_agf_nnet3(void* model_vp, char* grammar_fst_filename_cp);
extern "C" DRAGONFLY_API bool reload_grammar_fst_agf_nnet3(void* model_vp, int32_t grammar_fst_index, char* grammar_fst_filename_cp);
extern "C" DRAGONFLY_API bool remove_grammar_fst_agf_nnet3(void* model_vp, int32_t grammar_fst_index);
extern "C" DRAGONFLY_API bool decode_agf_nnet3(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize,
    bool* grammars_activity_cp, int32_t grammars_activity_cp_size, bool save_adaptation_state);
extern "C" DRAGONFLY_API bool get_output_agf_nnet3(void* model_vp, char* output, int32_t output_max_length,
    float* likelihood_p, float* am_score_p, float* lm_score_p, float* confidence_p, float* expected_error_rate_p);
