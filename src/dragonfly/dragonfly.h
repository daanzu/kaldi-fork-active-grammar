#pragma once

#ifdef DRAGONFLY_EXPORTS
#define DRAGONFLY_API __declspec(dllexport)
#else
#define DRAGONFLY_API __declspec(dllimport)
#endif

extern "C" DRAGONFLY_API int test();
