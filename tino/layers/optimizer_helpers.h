#pragma once

#include "tino/core/optimizers/optimizer.h"
#include "tino/utils/utils.h"

namespace tino {
  namespace layers {
    using namespace core;
    template <typename T>
    std::unique_ptr<core::optimizers::abstract_optimizer<T>> sgd(T alpha = 0.1) {
      TINO_MAYBE_UNUSED(alpha);
      return std::make_unique<core::optimizers::sgd_optimizer<T>>(alpha);
    }
  } // namespace layers
} // namespace tino