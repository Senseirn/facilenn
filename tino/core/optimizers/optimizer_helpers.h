#pragma once

#include "tino/core/optimizers/optimizer.h"
#include "tino/utils/utils.h"

namespace tino {
  namespace core {
    namespace optimizers {
      /*
      template <typename T = TINO_FLOAT_TYPE, typename U>
      std::unique_ptr<optimizers::abstract_optimizer_<T>> sgd(U alpha = 0.1) {
        TINO_MAYBE_UNUSED(alpha);
        return std::make_unique<optimizers::sgd_optimizer_<T>>(alpha);
      }
      */

      template <typename T>
      class SGD_;

      template <typename T>
      class Adam_;

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

        std::unique_ptr<optimizers::abstract_optimizer_<T>> operator()() const {
          return std::make_unique<optimizers::sgd_optimizer_<T>>(_alpha);
        }

        optimizer_t type() const { return optimizer_t::sgd; }
      };

      template <typename T = TINO_FLOAT_TYPE>
      class Adam_ {
       private:
        T _alpha;
        T _b1;
        T _b2;
        T _b1_t;
        T _b2_t;
        T _eps;

       public:
        Adam_(const T alpha = 0.01, const T b1 = 0.9, const T b2 = 0.999, const T eps = 1e-6)
        : _alpha(alpha)
        , _b1(b1)
        , _b2(b2)
        , _b1_t(b1)
        , _b2_t(b2)
        , _eps(eps) {}

        Adam_& alpha(const T alpha) {
          _alpha = alpha;
          return *this;
        }

        Adam_& b1(const T b1) {
          _b1 = b1;
          return *this;
        }

        Adam_& b2(const T b2) {
          _b2 = b2;
          return *this;
        }

        Adam_& eps(const T eps) {
          _eps = eps;
          return *this;
        }

        std::unique_ptr<optimizers::abstract_optimizer_<T>> operator()() const {
          return std::make_unique<optimizers::adam_optimizer_<T>>(_alpha, _b1, _b2, _eps);
        }

        optimizer_t type() const { return optimizer_t::adam; }
      };

      using SGD  = SGD_<TINO_FLOAT_TYPE>;
      using Adam = Adam_<TINO_FLOAT_TYPE>;
    } // namespace optimizers
  }   // namespace core
} // namespace tino