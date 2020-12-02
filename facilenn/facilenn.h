#pragma once

#include "backends/backends.h"
#include "core/core.h"
#include "layers/layers.h"
#include "models/models.h"
#include "utils/utils.h"

namespace fnn {
  template <typename T>
  class network {
   private:
    std::vector<std::unique_ptr<layers::abstract_layer<T>>> _net;

   public:
    void add(layers::abstract_layer<T>* layer) {
      _net.push_back(std::unique_ptr<layers::abstract_layer<T>>(layer));
    }

    void initialize(std::function<void(tensor2d<float>&)> initializer, std::size_t n_batch = 1) {
      for (auto& l : _net)
        l->initialize(initializer, n_batch);

      for (int i = 0; i < (int)_net.size(); i++)
        _net[i]->make_connection(i == 0 ? nullptr : _net[i - 1].get(),
                                 i == (int)_net.size() - 1 ? nullptr : _net[i + 1].get());
    }
  };
} // namespace fnn