#pragma once

#include "facilenn/backends/backend_types.h"
#include "facilenn/utils/utils.h"

namespace fnn {
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
} // namespace fnn