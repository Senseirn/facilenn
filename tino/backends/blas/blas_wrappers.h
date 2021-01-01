#pragma once

#include "tino/backends/backends.h"
#include "tino/backends/tensor_wrappers.h"
#include <cblas.h>

namespace tino::core {
  class context;
}

namespace tino {
  namespace backends {
    namespace blas {
      enum class layout_t { RowMajor = CblasRowMajor, ColumnMajor = CblasColMajor };
      enum class trans_t { NoTrans = CblasNoTrans, Trans = CblasTrans };

      template <typename T>
      struct blasOpts {
        T alpha = (T)0;
        T beta = (T)0;
        layout_t layout = layout_t::RowMajor;
        trans_t trans_a = trans_t::NoTrans;
        trans_t trans_b = trans_t::NoTrans;
      };

      template <typename T>
      tensor2d<T>&
      blas_gemm(core::context& ctx, const blasOpts<T>& blas_opts, tensor2d<T>& A, tensor2d<T>& B, tensor2d<T>& C) {}

      template <>
      tensor2d<float>& blas_gemm(core::context& ctx,
                                 const blasOpts<float>& blas_opts,
                                 tensor2d<float>& A,
                                 tensor2d<float>& B,
                                 tensor2d<float>& C) {

        cblas_sgemm(static_cast<CBLAS_LAYOUT>(blas_opts.layout),
                    static_cast<CBLAS_TRANSPOSE>(blas_opts.trans_a),
                    static_cast<CBLAS_TRANSPOSE>(blas_opts.trans_b),
                    A.template shape<1>(),
                    B.template shape<0>(),
                    A.template shape<0>(),
                    blas_opts.alpha,
                    A.data(),
                    A.template shape<0>(),
                    B.data(),
                    B.template shape<0>(),
                    blas_opts.beta,
                    C.data(),
                    C.template shape<0>());

        return C;
      }

    } // namespace blas
  }   // namespace backends
} // namespace tino