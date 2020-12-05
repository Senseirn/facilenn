#pragma once

#include "facilenn/backends/backends.h"
#include "facilenn/core/context.h"
#include "facilenn/utils/utils.h"

namespace fnn {
  namespace core {
    namespace op {
      using namespace fnn::core;
      using namespace fnn::backends;

      // forward declarations
      template <typename T>
      tensor2d<T>& fully_connected_forward_kernel_naive(tensor2d<T>& in,
                                                        tensor2d<T>& weight,
                                                        tensor2d<T>& bias,
                                                        tensor2d<T>& out,
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
          for (index_t j = 0; j < in.template shape<1>(); j++)
            for (index_t k = 0; k < in.template shape<0>(); k++)
              out(i, j) += in(i, k) * weight(k, j);

        for (index_t i = 0; i < in.template shape<1>(); i++)
          for (index_t j = 0; j < in.template shape<0>(); j++)
            out(i, j) += bias(0, j);

        FNN_MAYBE_UNUSED(ctx);

        return out;
      }

    } // namespace op
  }   // namespace core
} // namespace fnn