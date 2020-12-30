#pragma once

#include "tino/backends/backends.h"
#include "tino/core/core.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tino {
  namespace utils {
    // wrapper function for paralleled_for

    template <typename Index_t, typename F>
    void concurrent_for(tino::core::context& ctx, Index_t loop_upper_bound, F Lambda) {

/*
      constexpr std::size_t n_threads = 4;
      const std::size_t loops_per_thread = loop_upper_bound / n_threads;
      const std::size_t loop_remainder = loop_upper_bound % n_threads;
      */
#ifdef _OPENMP
#pragma omp parallel for
      for (Index_t i = 0; i < loop_upper_bound; i++) {
        Lambda(i);
      }
#else
      for (Index_t i = 0; i < loop_upper_bound; i++) {
        Lambda(i);
      }
#endif
      TINO_MAYBE_UNUSED(ctx);

      return;
    }

  } // namespace utils
} // namespace tino