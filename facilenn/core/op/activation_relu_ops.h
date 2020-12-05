#pragma once

#include "facilenn/backends/backends.h"
#include "facilenn/core/context.h"
#include "facilenn/utils/utils.h"

namespace fnn {
  namespace core {
    namespace op {
      using namespace fnn::core;
      using namespace fnn::backends;

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

        for (index_t i = 0; i < in.template shape<1>(); i++)
          for (index_t j = 0; j < in.template shape<0>(); j++)
            out(i, j) = in(i, j) > (T)0 ? in(i, j) : (T)0;

        FNN_MAYBE_UNUSED(ctx);

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

        for (index_t i = 0; i < in.template shape<1>(); i++)
          for (index_t j = 0; j < in.template shape<0>(); j++)
            delta(i, j) = (in(i, j) > (T)0 ? (T)1 : (T)0) * next_delta(i, j);

        FNN_MAYBE_UNUSED(ctx);

        return delta;
      }

    } // namespace op
  }   // namespace core
} // namespace fnn