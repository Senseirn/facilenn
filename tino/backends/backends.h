#pragma once

#include "backend_types.h"
#include "tino/backends/tensor/tensor/tensor.h"

namespace tino {
  template <typename T>
  using tensor1d = ts::tensor<T, 1>;
  template <typename T>
  using tensor2d = ts::tensor<T, 2>;
  template <typename T>
  using tensor3d = ts::tensor<T, 3>;
  template <typename T>
  using tensor4d = ts::tensor<T, 4>;
}