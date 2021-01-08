#pragma once

// check if OpenMP is availale.
// abort compilation when -fopenmp flag is not set.

#if not defined(TINO_PARALLEL_CHECKED)

#ifdef TINO_USE_OPENMP
#ifndef _OPENMP
#error TINO_USE_OPENMP is defined but OpenMP is not enabled. nplease re-compile with -fopenmp flag
#else
#define TINO_OPENMP_READY 1
//#include <omp.h>
#endif
#endif

// check if intel tbb is available.
#ifdef TINO_USE_INTEL_TBB
#define TINO_INTEL_TBB_READY 1

//#include <tbb/blocked_range.h>
//#include <tbb/parallel_for.h>
#endif

#define TINO_PARALLEL_CHECKED

#endif

#ifdef TINO_OPENMP_READY
#include <omp.h>
#endif

#ifdef TINO_INTEL_TBB_READY
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

// check backend
#ifndef TINO_BACKEND_CHECKED
#define TINO_BACKEND_CHECKED

#ifdef TINO_USE_OPENBLAS
#define TINO_OPENBLAS_READY 1
#endif
#endif

#include "tino/backends/blas/blas_wrappers.h"

namespace tino {
  namespace backends {
    enum class backend_t {
      naive,
      openblas,
    };

    enum class parallelize_t { none, openmp, intel_tbb };
  } // namespace backends
} // namespace tino