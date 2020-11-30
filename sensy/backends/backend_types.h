#pragma once

namespace ssy {
  namespace backends {
    enum class backend_t {
      naive,
      tensor,
      OpenBLAS,
      cuda, // naive cuda
      cublas
    };
  }
} // namespace ssy