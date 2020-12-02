#pragma once

#include "facilenn/backends/backends.h"
#include "facilenn/core/context.h"
#include "facilenn/utils/utils.h"

namespace fnn {
  namespace core {
    namespace optimzers {
      enum class optimizer_types { abstract, sgd, momentum, adagrad, adam };

      template <typename T>
      class abstract_optimizer {
       private:
        optimizer_types _optimizer_type;

       public:
        abstract_optimizer(optimizer_types optimizer_type)
        : _optimizer_type(optimizer_type) {}
        virtual optimizer_types type() { return _optimizer_type; }
        virtual optimize(tensor2d<T>&, tensor2d<T>&, context&) = 0;
      };

      template <typename T>
      class sgd : public abstract_optimizer<T> {
       private:
        const T _alpha; // learning rate
       public:
        sgd(T alpha = 0.1)
        : abstract_optimizer<T>(optimizer_types::sgd) {}
        optimize(tensor2d<T>& weight, tensor2d<T>& delta, context& ctx) {}
      };
    } // namespace optimzers
  }   // namespace core
} // namespace fnn