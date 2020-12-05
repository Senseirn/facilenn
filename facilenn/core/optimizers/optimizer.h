#pragma once

#include "facilenn/backends/backends.h"
#include "facilenn/core/context.h"
#include "facilenn/utils/utils.h"

namespace fnn {
  namespace core {
    namespace optimizers {
      enum class optimizers { abstract, sgd, momentum, adagrad, adam };

      template <typename T>
      class abstract_optimizer {
       private:
        optimizers _optimizer_type;

       public:
        abstract_optimizer(optimizers optimizer_type)
        : _optimizer_type(optimizer_type) {}
        virtual optimizers type() { return _optimizer_type; }
        virtual void optimize(tensor2d<T>&, tensor2d<T>&, context&) = 0;
      };

      template <typename T>
      class sgd_optimizer : public abstract_optimizer<T> {
       private:
        const T _alpha; // learning rate
       public:
        sgd_optimizer(T alpha = 0.1)
        : abstract_optimizer<T>(optimizers::sgd)
        , _alpha(alpha) {}
        void optimize(tensor2d<T>& weight, tensor2d<T>& delta, context& ctx) override {
          FNN_MAYBE_UNUSED(weight);
          FNN_MAYBE_UNUSED(delta);
          FNN_MAYBE_UNUSED(ctx);
        }
      };

    } // namespace optimizers
  }   // namespace core
} // namespace fnn