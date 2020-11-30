#pragma once

#include "facilenn/backends/tensor/tensor_core.h"

namespace fnn {
  template <typename T>
  using tensor2d = ts::tensor<T, 2>;
}