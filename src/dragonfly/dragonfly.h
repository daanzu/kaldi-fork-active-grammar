#pragma once

#ifdef DRAGONFLY_EXPORTS
#define DRAGONFLY_API __declspec(dllexport)
#else
#define DRAGONFLY_API __declspec(dllimport)
#endif

#include <stdint.h>
//#include "targetver.h"
//#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
//#include <windows.h>

//extern "C" DRAGONFLY_API int test();

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

extern "C" DRAGONFLY_API void* init_agf_nnet3(float beam, int32_t max_active, int32_t min_active, float lattice_beam, float acoustic_scale, int32_t frame_subsampling_factor,
    int32_t nonterm_phones_offset, char* word_syms_filename_cp, char* mfcc_config_filename_cp, char* ie_config_filename_cp,
    char* model_filename_cp, char* top_fst_filename_cp);
extern "C" DRAGONFLY_API bool add_grammar_fst_agf_nnet3(void* model_vp, char* grammar_fst_filename_cp);
extern "C" DRAGONFLY_API bool decode_agf_nnet3(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize,
    bool* grammars_activity_cp, int32_t grammars_activity_cp_size, bool save_adaptation_state);
extern "C" DRAGONFLY_API bool get_output_agf_nnet3(void* model_vp, char* output, int32_t output_length, double* likelihood_p);
extern "C" DRAGONFLY_API void reset_adaptation_state_agf_nnet3(void* model_vp);
