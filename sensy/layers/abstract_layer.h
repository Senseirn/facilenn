#pragma once

#include "sensy/backends/backends.h"
#include "sensy/utils/utils.h"

namespace ssy {
  namespace layers {
    template <typename T>
    class abstract_layer {
     private:
      tensor2d<float> _in;
    };
  } // namespace layers
} // namespace ssy