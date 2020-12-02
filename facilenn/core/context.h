#pragma once

#include "facilenn/backends/backend_types.h"
#include "facilenn/utils/utils.h"

namespace fnn {
  namespace core {
    enum class stages { train, infer };
    class context {
     private:
      backends::backend_t _backend;
      stages _stage;

     public:
      context(const backends::backend_t& backend, stages stage = stages::train)
      : _backend(backend)
      , _stage(stage) {}
      backends::backend_t backend() { return _backend; }
      backends::backend_t backend() const { return _backend; }
      context& set_backend(const backends::backend_t& backend) {
        _backend = backend;
        return *this;
      }

      bool is_train() { return _stage == stages::train; }
      bool is_infer() { return !is_train(); }
      stages current_stage() { return _stage; }
      context& set_stage(const stages stage) {
        _stage = stage;
        return *this;
      }
    };
  } // namespace core
} // namespace fnn