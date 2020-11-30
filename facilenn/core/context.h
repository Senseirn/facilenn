#pragma once

#include "sensy/backends/backend_types.h"
#include "sensy/utils/utils.h"

namespace ssy {
  namespace core {
    class context {
     private:
      backends::backend_t _backend;

     public:
      context(const backends::backend_t& backend)
      : _backend(backend) {}
      backends::backend_t backend() { return _backend; }
      backends::backend_t backend() const { return _backend; }
      context& set_backend(const backends::backend_t& backend) {
        _backend = backend;
        return *this;
      }
    };
  } // namespace core
} // namespace ssy