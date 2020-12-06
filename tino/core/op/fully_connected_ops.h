#pragma once

#include "tino/backends/backends.h"
#include "tino/core/context.h"
#include "tino/utils/utils.h"

namespace tino {
  namespace core {
    namespace op {
      using namespace tino::core;
      using namespace tino::backends;

      // forward declarations
      template <typename T>
      tensor2d<T>& fully_connected_forward_kernel_naive(tensor2d<T>& in,
                                                        tensor2d<T>& weight,
                                                        tensor2d<T>& bias,
                                                        tensor2d<T>& out,
                                                        context& ctx);

      template <typename T>
      tensor2d<T>& fully_connected_backward_kernel_naive(tensor2d<T>& in,
                                                         tensor2d<T>& next_delta,
                                                         tensor2d<T>& weight,
                                                         tensor2d<T>& delta,
                                                         tensor2d<T>& delta_weight,
                                                         tensor2d<T>& delta_bias,
                                                         context& ctx);

      // actual implementations
      template <typename T>
      tensor2d<T>& fully_connected_forward_kernel(tensor2d<T>& in,
                                                  tensor2d<T>& weight,
                                                  tensor2d<T>& bias,
                                                  tensor2d<T>& out,
                                                  context& ctx) {
        if (ctx.backend() == backend_t::naive)
          return fully_connected_forward_kernel_naive(in, weight, bias, out, ctx);

        return out;
      }

      template <typename T>
      tensor2d<T>& fully_connected_forward_kernel_naive(tensor2d<T>& in,
                                                        tensor2d<T>& weight,
                                                        tensor2d<T>& bias,
                                                        tensor2d<T>& out,
                                                        context& ctx) {
        std::fill(std::begin(out), std::end(out), (T)0);

        using index_t = std::size_t;
        for (index_t i = 0; i < in.template shape<1>(); i++)
          for (index_t j = 0; j < weight.template shape<1>(); j++)
            for (index_t k = 0; k < in.template shape<0>(); k++)
              out(i, j) += in(i, k) * weight(k, j);

        for (index_t i = 0; i < out.template shape<1>(); i++)
          for (index_t j = 0; j < out.template shape<0>(); j++)
            out(i, j) += bias(0, j);

        TINO_MAYBE_UNUSED(ctx);

        return out;
      }

      template <typename T>
      tensor2d<T>& fully_connected_backward_kernel(tensor2d<T>& in,
                                                   tensor2d<T>& next_delta,
                                                   tensor2d<T>& weight,
                                                   tensor2d<T>& delta,
                                                   tensor2d<T>& delta_weight,
                                                   tensor2d<T>& delta_bias,
                                                   context& ctx) {
        if (ctx.backend() == backend_t::naive)
          return fully_connected_backward_kernel_naive(in, next_delta, weight, delta, delta_weight, delta_bias, ctx);

        return delta;
      }

      template <typename T>
      tensor2d<T>& fully_connected_backward_kernel_naive(tensor2d<T>& in,
                                                         tensor2d<T>& next_delta,
                                                         tensor2d<T>& weight,
                                                         tensor2d<T>& delta,
                                                         tensor2d<T>& delta_weight,
                                                         tensor2d<T>& delta_bias,
                                                         context& ctx) {
        std::fill(std::begin(delta), std::end(delta), (T)0);
        std::fill(std::begin(delta_weight), std::end(delta_weight), (T)0);

        using index_t = std::size_t;
        for (index_t i = 0; i < next_delta.template shape<1>(); i++)
          for (index_t j = 0; j < weight.template shape<1>(); j++)
            for (index_t k = 0; k < next_delta.template shape<0>(); k++)
              delta(i, j) += next_delta(i, k) * weight(j, k);

        for (index_t i = 0; i < in.template shape<0>(); i++)
          for (index_t j = 0; j < next_delta.template shape<0>(); j++)
            for (index_t k = 0; k < in.template shape<1>(); k++)
              delta_weight(i, j) += in(k, i) * next_delta(k, j);

        for (index_t i = 0; i < delta_weight.template shape<1>(); i++)
          for (index_t j = 0; j < delta_weight.template shape<0>(); j++)
            delta_weight(i, j) /= in.template shape<1>(); // div by n_batch.

        TINO_MAYBE_UNUSED(delta_bias);
        TINO_MAYBE_UNUSED(ctx);

        return delta;
      }

    } // namespace op
  }   // namespace core
} // namespace tino