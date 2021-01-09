#pragma once

#include "tino/backends/backends.h"
#include "tino/core/context.h"
#include "tino/utils/utils.h"

namespace tino {
  namespace core {
    namespace optimizers {
      enum class optimizer_t { none, abstract, sgd, momentum, adagrad, adam };

      template <typename T>
      class abstract_optimizer_ {
       private:
        optimizer_t _optimizer_type;

       public:
        abstract_optimizer_(optimizer_t optimizer_type)
        : _optimizer_type(optimizer_type) {}
        virtual void init(std::size_t, std::size_t) {}
        virtual optimizer_t type() { return _optimizer_type; }
        virtual void optimize(tensor2d<T>&, tensor2d<T>&, context&) = 0;
      };

      template <typename T = TINO_FLOAT_TYPE>
      class sgd_optimizer_ : public abstract_optimizer_<T> {
       private:
        const T _alpha; // learning rate
       public:
        sgd_optimizer_(T alpha = 0.1)
        : abstract_optimizer_<T>(optimizer_t::sgd)
        , _alpha(alpha) {}

        void optimize(tensor2d<T>& weight, tensor2d<T>& delta_weight, context& ctx) override {
          using index_t = std::size_t;
          for (index_t i = 0; i < weight.template shape<1>(); i++)
            for (index_t j = 0; j < weight.template shape<0>(); j++)
              weight(i, j) -= _alpha * delta_weight(i, j);

          TINO_MAYBE_UNUSED(ctx);
        }
      };

      template <typename T = TINO_FLOAT_TYPE>
      class adam_optimizer_ : public abstract_optimizer_<T> {
       private:
        const T _alpha;
        const T _b1;
        const T _b2;
        T _b1_t;
        T _b2_t;
        const T _eps;

        tensor2d<T> _mt;
        tensor2d<T> _vt;

       public:
        adam_optimizer_(T alpha = 0.001, T b1 = 0.9, T b2 = 0.999, T eps = 1e-6)
        : abstract_optimizer_<T>(optimizer_t::adam)
        , _alpha(alpha)
        , _b1(b1)
        , _b2(b2)
        , _b1_t(b1)
        , _b2_t(b2)
        , _eps(eps) {}

        void init(std::size_t d1, std::size_t d0) override {
          _mt.reshape(d1, d0);
          _vt.reshape(d1, d0);
        }

        void optimize(tensor2d<T>& weight, tensor2d<T>& delta_weight, context& ctx) override {
          using index_t = std::size_t;

          utils::concurrent_for(ctx, weight.template shape<1>(), [&](index_t i) {
            for (index_t j = 0; j < weight.template shape<0>(); j++) {
              _mt(i, j) = _b1 * _mt(i, j) + ((T)1 - _b1) * delta_weight(i, j);
              _vt(i, j) = _b2 * _vt(i, j) + ((T)1 - _b2) * delta_weight(i, j) * delta_weight(i, j);

              weight(i, j) -= _alpha * (_mt(i, j) / ((T)1 - _b1_t)) / (std::sqrt((_vt(i, j) / ((T)1 - _b2_t))) + _eps);

              _b1_t *= _b1;
              _b2_t *= _b2;
            }
          });

          TINO_MAYBE_UNUSED(ctx);
        }
      };

    } // namespace optimizers
  }   // namespace core
} // namespace tino