#pragma once

#include "facilenn/backends/tensor/tensor_core.h"

namespace fnn {
  template <typename T>
  using tensor1d = ts::tensor<T, 1>;
  template <typename T>
  using tensor2d = ts::tensor<T, 2>;

  template <typename T>
  using tensor3d = ts::tensor<T, 3>;

  template <typename T>
  using tensor4d = ts::tensor<T, 4>;
} // namespace fnn