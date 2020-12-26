#pragma once

#include "tino/core/optimizers/optimizer.h"
#include "tino/utils/utils.h"

namespace tino {
  namespace core {
    /*
    template <typename T = TINO_FLOAT_TYPE, typename U>
    std::unique_ptr<optimizers::abstract_optimizer_<T>> sgd(U alpha = 0.1) {
      TINO_MAYBE_UNUSED(alpha);
      return std::make_unique<optimizers::sgd_optimizer_<T>>(alpha);
    }
    */

    template <typename T>
    class SGD_;

    template <typename T = TINO_FLOAT_TYPE>
    class SGD_ {
     private:
      T _alpha;

     public:
      SGD_(const T alpha = 0.1)
      : _alpha(alpha) {}

      SGD_& alpha(const T alpha) {
        _alpha = alpha;
        return *this;
      }

      std::unique_ptr<optimizers::abstract_optimizer_<T>> operator()() {
        return std::make_unique<optimizers::sgd_optimizer_<T>>(_alpha);
      }
    };
    using SGD = SGD_<TINO_FLOAT_TYPE>;
  } // namespace core
} // namespace tino