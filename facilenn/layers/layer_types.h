#pragma once

namespace fnn {
  namespace layers {
    enum class layer_types {
      abstract,
      fully_connected,
      flatten,
      act_relu,
      act_sigmoid,
      act_softmax
    };
  }
} // namespace fnn