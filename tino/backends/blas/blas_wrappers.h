#pragma once

#include "tino/backends/backends.h"
#include "tino/backends/tensor_wrappers.h"
#include "tino/utils/macros.h"

#ifdef TINO_OPENBLAS_READY
#include <cblas.h>
#endif

namespace tino {
  namespace core {
    class context;
  }
  namespace backends {
    namespace blas {
      enum class layout_t {
#ifdef TINO_OPENBLAS_READY
        RowMajor    = CblasRowMajor,
        ColumnMajor = CblasColMajor
#else
        RowMajor,
        ColumnMajor
#endif
      };
      enum class trans_t {
#ifdef TINO_OPENBLAS_READY
        NoTrans = CblasNoTrans,
        Trans   = CblasTrans
#else
        NoTrans,
        Trans
#endif
      };

      template <typename T>
      struct blasOpts {
        T alpha         = (T)0;
        T beta          = (T)0;
        layout_t layout = layout_t::RowMajor;
        trans_t trans_a = trans_t::NoTrans;
        trans_t trans_b = trans_t::NoTrans;
      };

      template <typename T>
      tensor2d<T>& blas_gemm(core::context& ctx, const blasOpts<T>& blas_opts, tensor2d<T>& A, tensor2d<T>& B, tensor2d<T>& C) {}

      template <>
      tensor2d<float>& blas_gemm(core::context& ctx, const blasOpts<float>& blas_opts, tensor2d<float>& A, tensor2d<float>& B, tensor2d<float>& C) {
#ifdef TINO_OPENBLAS_READY
        const int M   = blas_opts.trans_a == trans_t::NoTrans ? A.template shape<1>() : A.template shape<0>();
        const int N   = blas_opts.trans_b == trans_t::NoTrans ? B.template shape<0>() : B.template shape<1>();
        const int K   = blas_opts.trans_a == trans_t::NoTrans ? A.template shape<0>() : A.template shape<1>();
        const int lda = blas_opts.trans_a == trans_t::NoTrans ? K : M;
        const int ldb = blas_opts.trans_b == trans_t::NoTrans ? N : K;
        const int ldc = N;

        cblas_sgemm(static_cast<CBLAS_LAYOUT>(blas_opts.layout),
                    static_cast<CBLAS_TRANSPOSE>(blas_opts.trans_a),
                    static_cast<CBLAS_TRANSPOSE>(blas_opts.trans_b),
                    M,
                    N,
                    K,
                    blas_opts.alpha,
                    A.data(),
                    lda,
                    B.data(),
                    ldb,
                    blas_opts.beta,
                    C.data(),
                    ldc);
#else
        std::cerr << "invalid backend: OpenBLAS" << std::endl;
        std::exit(1);
#endif

        TINO_MAYBE_UNUSED(ctx);
        TINO_MAYBE_UNUSED(blas_opts);
        TINO_MAYBE_UNUSED(A);
        TINO_MAYBE_UNUSED(B);
        TINO_MAYBE_UNUSED(C);

        return C;
      }

    } // namespace blas
  }   // namespace backends
} // namespace tino