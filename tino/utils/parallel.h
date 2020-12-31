#pragma once

// check if OpenMP is availale.
// abort compilation when -fopenmp flag is not set.
#ifdef TINO_USE_OPENMP
#ifndef _OPENMP
#error TINO_USE_OPENMP is defined but OpenMP is not enabled. nplease re-compile with -fopenmp flag
#else
#define TINO_OPENMP_READY 1
#include <omp.h>
#endif
#endif

// check if intel tbb is available.
#ifdef TINO_USE_INTEL_TBB
#define TINO_INTEL_TBB_READY 1

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

/*
// if both of OpenMP and intel tbb define, abort!
#if defined(TINO_OPENMP_READY) && defined(TINO_INTEL_TBB_READY)
#error parallel-backend is exclusive each other.
#endif
*/

#include "tino/backends/backends.h"
#include "tino/core/core.h"

namespace tino {
  namespace utils {
    // wrapper function for paralleled_for

    template <typename Index_t, typename F>
    void concurrent_for(tino::core::context& ctx, Index_t loop_upper_bound, F Lambda) {

      switch (ctx.parallelize()) {
        case backends::parallelize_t::none:
          for (Index_t i = 0; i < loop_upper_bound; i++) {
            Lambda(i);
          }
          break;
        case backends::parallelize_t::openmp:
#if defined(TINO_OPENMP_READY)
#pragma omp parallel for
          for (Index_t i = 0; i < loop_upper_bound; i++) {
            Lambda(i);
          }
#else
          std::cerr << "invalild parallel-backend: OpenMP" << std::endl;
          std::exit(1);
#endif
          break;
        case backends::parallelize_t::intel_tbb:
#if defined(TINO_INTEL_TBB_READY)
          tbb::parallel_for(tbb::blocked_range<int>(0, loop_upper_bound), [&](const tbb::blocked_range<int>& r) {
            for (int list = r.begin(); list < r.end(); list++) {
              Lambda(list);
            }
          });
#else
          std::cerr << "invalild parallel-backend: Intel TBB" << std::endl;
          std::exit(1);
#endif
          break;
        default: break;
      }
      TINO_MAYBE_UNUSED(ctx);

      return;
    }

  } // namespace utils
} // namespace tino