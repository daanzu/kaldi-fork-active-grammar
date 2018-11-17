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

extern "C" DRAGONFLY_API int test();

extern "C" DRAGONFLY_API void* init_gmm(float beam, int32_t max_active, int32_t min_active, float lattice_beam,
    char* word_syms_filename_cp, char* fst_in_str_cp, char* config_cp);
extern "C" DRAGONFLY_API bool decode_gmm(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize);
extern "C" DRAGONFLY_API void get_output_gmm(void* model_vp, char* output, int32_t output_length, double* likelihood_p);
