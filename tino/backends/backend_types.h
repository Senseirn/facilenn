#pragma once

namespace tino {
  namespace backends {
    enum class backend_t {
      naive,
      OpenBLAS,
    };

    enum class prallelize_t { none, openmp, tbb };
  } // namespace backends
} // namespace tino