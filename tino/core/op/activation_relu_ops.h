#pragma once

#include "tino/backends/backends.h"
#include "tino/core/context.h"
#include "tino/utils/utils.h"

namespace tino {
  namespace core {
    namespace op {
      using namespace tino::core;
      using namespace tino::backends;

      template <typename T>
      tensor2d<T>& relu_activation_forward_kernel_naive(tensor2d<T>& in, tensor2d<T>& out, context& ctx);

      template <typename T>
      tensor2d<T>&
      relu_activation_backward_kernel_naive(tensor2d<T>& in, tensor2d<T>& delta, tensor2d<T>& next_delta, context& ctx);

      // actual implementations
      template <typename T>
      tensor2d<T>& relu_activation_forward_kernel(tensor2d<T>& in, tensor2d<T>& out, context& ctx) {
        if (ctx.backend() == backend_t::naive)
          return relu_activation_forward_kernel_naive(in, out, ctx);

        return out;
      }

      template <typename T>
      tensor2d<T>& relu_activation_forward_kernel_naive(tensor2d<T>& in, tensor2d<T>& out, context& ctx) {
        using index_t = std::size_t;

        utils::concurrent_for(ctx, in.template shape<1>(), [&](index_t i) {
          for (index_t j = 0; j < in.template shape<0>(); j++)
            out(i, j) = in(i, j) > (T)0 ? in(i, j) : (T)0;
        });

        TINO_MAYBE_UNUSED(ctx);

        return out;
      }

      template <typename T>
      tensor2d<T>&
      relu_activation_backward_kernel(tensor2d<T>& in, tensor2d<T>& delta, tensor2d<T>& next_delta, context& ctx) {
        if (ctx.backend() == backend_t::naive)
          return relu_activation_backward_kernel_naive(in, delta, next_delta, ctx);

        return delta;
      }

      template <typename T>
      tensor2d<T>& relu_activation_backward_kernel_naive(tensor2d<T>& in,
                                                         tensor2d<T>& delta,
                                                         tensor2d<T>& next_delta,
                                                         context& ctx) {
        using index_t = std::size_t;
        utils::concurrent_for(ctx, in.template shape<1>(), [&](index_t i) {
          for (index_t j = 0; j < in.template shape<0>(); j++)
            delta(i, j) = (in(i, j) > (T)0 ? next_delta(i, j) : (T)0);
        });

        TINO_MAYBE_UNUSED(ctx);

        return delta;
      }

    } // namespace op
  }   // namespace core
} // namespace tino