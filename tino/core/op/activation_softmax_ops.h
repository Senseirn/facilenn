#pragma once

#include "tino/backends/backends.h"
#include "tino/core/context.h"
#include "tino/utils/utils.h"

#include <cmath>

namespace tino {
  namespace core {
    namespace op {
      using namespace tino::core;
      using namespace tino::backends;

      template <typename T>
      tensor2d<T>& softmax_activation_forward_kernel_naive(tensor2d<T>& in, tensor2d<T>& out, context& ctx);

      template <typename T>
      tensor2d<T>& softmax_activation_backward_kernel_naive(tensor2d<T>& in,
                                                            tensor2d<T>& delta,
                                                            tensor2d<T>& next_delta,
                                                            context& ctx);

      // actual implementations
      template <typename T>
      tensor2d<T>& softmax_activation_forward_kernel(tensor2d<T>& in, tensor2d<T>& out, context& ctx) {
        if (ctx.backend() == backend_t::naive)
          return softmax_activation_forward_kernel_naive(in, out, ctx);

        return out;
      }

      template <typename T>
      tensor2d<T>& softmax_activation_forward_kernel_naive(tensor2d<T>& in, tensor2d<T>& out, context& ctx) {
        using index_t = std::size_t;

        for (index_t i = 0; i < in.template shape<1>(); i++) {
          T accum = (T)0;
          T maximum = std::numeric_limits<T>::lowest();
          for (index_t j = 0; j < in.template shape<0>(); j++)
            maximum = std::max(maximum, in(i, j));

          for (index_t j = 0; j < in.template shape<0>(); j++)
            accum += std::exp(in(i, j) - maximum);

          for (index_t j = 0; j < in.template shape<0>(); j++)
            out(i, j) = std::exp(in(i, j) - maximum) / accum;
        }

        TINO_MAYBE_UNUSED(ctx);

        return out;
      }

      template <typename T>
      tensor2d<T>&
      softmax_activation_backward_kernel(tensor2d<T>& in, tensor2d<T>& delta, tensor2d<T>& next_delta, context& ctx) {
        if (ctx.backend() == backend_t::naive)
          return softmax_activation_backward_kernel_naive(in, delta, next_delta, ctx);

        return delta;
      }

      template <typename T>
      tensor2d<T>& softmax_activation_backward_kernel_naive(tensor2d<T>& in,
                                                            tensor2d<T>& delta,
                                                            tensor2d<T>& next_delta,
                                                            context& ctx) {

        TINO_MAYBE_UNUSED(in);
        TINO_MAYBE_UNUSED(delta);
        TINO_MAYBE_UNUSED(next_delta);
        TINO_MAYBE_UNUSED(ctx);

        return delta;
      }

    } // namespace op
  }   // namespace core
} // namespace tino