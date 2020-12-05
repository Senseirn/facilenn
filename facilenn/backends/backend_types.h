#pragma once

namespace fnn {
  namespace backends {
    enum class backend_t {
      naive,
      tensor,
      OpenBLAS,
      cuda, // naive cuda
      cublas
    };

    enum class prallelize_t { none, openmp, tbb };
  } // namespace backends
} // namespace fnn