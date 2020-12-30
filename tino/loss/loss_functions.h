#pragma once

#include "tino/backends/backends.h"
#include "tino/core/core.h"
#include "tino/utils/utils.h"
#include <cfloat>

namespace tino {
  namespace loss {
    template <typename T>
    class abstract_loss {};

    template <typename T>
    class mse : public abstract_loss<T> {
     public:
      static T f(tensor2d<T>& y, tensor2d<T>& t, core::context& ctx) {
        using index_t = typename tensor2d<T>::index_t;
        T loss = 0;
        for (index_t i = 0; i < y.template shape<1>(); i++) {
          for (index_t j = 0; j < y.template shape<0>(); j++) {
            loss += std::pow(y(i, j) - t(i, j), 2);
          }
        }

        TINO_MAYBE_UNUSED(ctx);

        return loss;
      }

      static tensor2d<T>& df(tensor2d<T>& y, tensor2d<T>& t, tensor2d<T>& error, core::context& ctx) {
        using index_t = typename tensor2d<T>::index_t;
        for (index_t i = 0; i < y.template shape<1>(); i++) {
          for (index_t j = 0; j < y.template shape<0>(); j++) {
            error(i, j) = y(i, j) - t(i, j);
          }
        }
        TINO_MAYBE_UNUSED(ctx);
        return error;
      }
    };

    template <typename T>
    class cross_entropy : public abstract_loss<T> {
     public:
      static T f(tensor2d<T>& y, tensor2d<T>& t, core::context& ctx) {
        using index_t = typename tensor2d<T>::index_t;
        T loss = 0;
        for (index_t i = 0; i < y.template shape<1>(); i++) {
          for (index_t j = 0; j < y.template shape<0>(); j++) {
            loss += t(i, j) * std::log(FLT_MIN + y(i, j));
          }
        }

        TINO_MAYBE_UNUSED(ctx);

        return -loss;
      }

      static tensor2d<T>& df(tensor2d<T>& y, tensor2d<T>& t, tensor2d<T>& error, core::context& ctx) {
        using index_t = typename tensor2d<T>::index_t;
        for (index_t i = 0; i < y.template shape<1>(); i++) {
          for (index_t j = 0; j < y.template shape<0>(); j++) {
            error(i, j) = y(i, j) - t(i, j);
          }
        }
        TINO_MAYBE_UNUSED(ctx);
        return error;
      }
    };
  } // namespace loss
} // namespace tino