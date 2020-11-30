#pragma once

#include "facilenn/backends/backends.h"
#include "facilenn/utils/utils.h"

namespace fnn {
  namespace layers {
    template <typename T>
    class abstract_layer {
     private:
      tensor2d<float> _in;
    };
  } // namespace layers
} // namespace fnn