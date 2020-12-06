#pragma once

#include "tino/backends/backends.h"
#include "tino/core/context.h"
#include "tino/utils/utils.h"

namespace tino {
  namespace core {
    namespace optimizers {
      enum class optimizer_t { abstract, sgd, momentum, adagrad, adam };

      template <typename T>
      class abstract_optimizer {
       private:
        optimizer_t _optimizer_type;

       public:
        abstract_optimizer(optimizer_t optimizer_type)
        : _optimizer_type(optimizer_type) {}
        virtual optimizer_t type() { return _optimizer_type; }
        virtual void optimize(tensor2d<T>&, tensor2d<T>&, context&) = 0;
      };

      template <typename T>
      class sgd_optimizer : public abstract_optimizer<T> {
       private:
        const T _alpha; // learning rate
       public:
        sgd_optimizer(T alpha = 0.1)
        : abstract_optimizer<T>(optimizer_t::sgd)
        , _alpha(alpha) {}

        void optimize(tensor2d<T>& weight, tensor2d<T>& delta_weight, context& ctx) override {
          using index_t = std::size_t;
          for (index_t i = 0; i < weight.template shape<1>(); i++)
            for (index_t j = 0; j < weight.template shape<0>(); j++)
              weight(i, j) -= _alpha * delta_weight(i, j);

          TINO_MAYBE_UNUSED(ctx);
        }
      };

    } // namespace optimizers
  }   // namespace core
} // namespace tino