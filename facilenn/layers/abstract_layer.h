#pragma once

#include "facilenn/backends/backends.h"
#include "facilenn/utils/utils.h"

namespace fnn {
  namespace layers {
    template <typename T>
    class abstract_layer;
    template <typename T>
    class abstract_layer {
     private:
      tensor2d<T> _in;
      tensor2d<T> _out;
      tensor2d<T> _weight;
      tensor2d<T> _delta;

      std::shared_ptr<abstract_layer<T>> _prev_layer;
      std::shared_ptr<abstract_layer<T>> _next_layer;

      std::size_t _in_size;
      std::size_t _out_size;
    };
  } // namespace layers
} // namespace fnn