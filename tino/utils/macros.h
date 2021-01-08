#pragma once

// supress warning for unused vars
#define TINO_MAYBE_UNUSED(x) (void)(x)

#ifndef TINO_FLOAT_TYPE
#define TINO_FLOAT_TYPE float
#endif

#include "tino/backends/backend_types.h"

namespace tino {
  namespace utils {
    static inline void backend_availability() {
#if not defined(TINO_BACKEND_CHECKED)
#define TINO_BACKEND_CHECKED

#endif
    }

    static inline void parallelize_availavility() {
#if not defined(TINO_PARALLEL_CHECKED)
#define TINO_PARALLEL_CHECKED

#ifdef TINO_USE_OPENMP
#ifndef _OPENMP
#error TINO_USE_OPENMP is defined but OpenMP is not enabled. nplease re-compile with -fopenmp flag
#else
#define TINO_OPENMP_READY 1
      std::cout << "openmp available" << std::endl;

#endif
#endif

#ifdef TINO_USE_INTEL_TBB
#define TINO_INTEL_TBB_READY 1
      std::cout << "intel tbb available" << std::endl;
#endif

#endif
    }
  } // namespace utils
} // namespace tino