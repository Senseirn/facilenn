#pragma once

#include "tino/backends/backend_types.h"

namespace tino {
  namespace core {
    enum class stages { train, infer };
    class context {
     private:
      backends::backend_t _backend;
      backends::parallelize_t _parallel;
      stages _stage;

     public:
      context(const backends::backend_t& backend      = backends::backend_t::naive,
              const backends::parallelize_t& parallel = backends::parallelize_t::none,
              const stages& stage                     = stages::train)
      : _backend(backend)
      , _parallel(parallel)
      , _stage(stage) {}

      backends::backend_t backend() { return _backend; }
      backends::backend_t backend() const { return _backend; }
      context& backend(backends::backend_t backend) {
        _backend = backend;
        return *this;
      }
      context& set_backend(const backends::backend_t& backend) {
        _backend = backend;
        return *this;
      }

      backends::parallelize_t parallelize() { return _parallel; }
      backends::parallelize_t parallelize() const { return _parallel; }
      context& parallelize(backends::parallelize_t parallel) {
        _parallel = parallel;
        return *this;
      }

      bool is_train() { return _stage == stages::train; }
      bool is_infer() { return !is_train(); }
      stages current_stage() { return _stage; }
      context& stage(const stages stage) {
        _stage = stage;
        return *this;
      }
      context& set_stage(const stages stage) {
        _stage = stage;
        return *this;
      }
    };
  } // namespace core
} // namespace tino