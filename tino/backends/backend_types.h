#pragma once

namespace tino {
  namespace backends {
    enum class backend_t {
      naive,
      openblas,
    };

    enum class parallelize_t { none, openmp, intel_tbb };
  } // namespace backends
} // namespace tino