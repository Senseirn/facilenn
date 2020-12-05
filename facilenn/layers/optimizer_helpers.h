#pragma once

#include "facilenn/core/optimizers/optimizer.h"
#include "facilenn/utils/utils.h"

namespace fnn {
  namespace layers {
    using namespace core;
    template <typename T>
    std::unique_ptr<core::optimizers::abstract_optimizer<T>> sgd(T alpha = 0.1) {
      return std::make_unique<optimizers::sgd_optimizer<T>>(alpha);
    }
  } // namespace layers
} // namespace fnn