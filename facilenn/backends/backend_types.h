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
  }
} // namespace fnn