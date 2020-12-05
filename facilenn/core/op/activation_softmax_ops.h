#pragma once

#include "facilenn/backends/backends.h"
#include "facilenn/core/context.h"
#include "facilenn/utils/utils.h"

#include "cmath"

namespace fnn {
  namespace core {
    namespace op {
      using namespace fnn::core;
      using namespace fnn::backends;

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

          for (index_t j = 0; j < in.template shape<0>(); j++) {
            out(i, j) = std::exp(in(i, j) - maximum) / accum;
          }
        }

        FNN_MAYBE_UNUSED(ctx);

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

        FNN_MAYBE_UNUSED(in);
        FNN_MAYBE_UNUSED(delta);
        FNN_MAYBE_UNUSED(next_delta);
        FNN_MAYBE_UNUSED(ctx);

        return delta;
      }

    } // namespace op
  }   // namespace core
} // namespace fnn