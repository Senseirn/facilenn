#pragma once

#include "sensy/backends/tensor/tensor_core.h"

namespace ssy {
  template <typename T>
  using tensor2d = ts::tensor<T, 2>;
}