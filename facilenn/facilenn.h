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
    bool _is_initialized = false;

    void prepare_batched_data(tensor2d<T>& inputs,
                              tensor2d<T>& labels,
                              std::vector<tensor2d<T>>& batched_inputs,
                              std::vector<tensor2d<T>>& batched_labels,
                              std::size_t n_minibatchs) {
      std::size_t batchs = inputs.template shape<1>() / n_minibatchs;
      batched_inputs.resize(batchs);
      batched_labels.resize(batchs);
      for (std::size_t i = 0; i < batchs; i++) {
        batched_inputs[i]->reshape(n_minibatchs, inputs.template shape<0>());
        batched_labels[i]->reshape(n_minibatchs, labels.template shape<0>());
        for (std::size_t j = 0; j < n_minibatchs; j++) {
          for (std::size_t k = 0; k < inputs.template shape<0>(); k++)
            batched_inputs[i](j, k) = inputs(i * n_minibatchs + j, k);
          for (std::size_t k = 0; k < labels.template shape<0>(); k++)
            batched_labels[i](j, k) = labels(i * n_minibatchs, k++);
        }
      }
    }

    void forward_prop(tensor2d<T>& input, tensor2d<T>& label, core::context& ctx) {
      _net.front()->forward(input, label, ctx);
    }

   public:
    void add(layers::abstract_layer<T>* layer) { _net.push_back(std::unique_ptr<layers::abstract_layer<T>>(layer)); }

    bool initialize(
        std::size_t n_batch = 1,
        std::function<void(tensor2d<float>&)> initializer = [](tensor2d<float>& x) {
          for (auto& e : x)
            e = (float)0.1;
        }) {
      for (auto& l : _net)
        l->initialize(initializer, n_batch);

      for (int i = 0; i < (int)_net.size(); i++)
        _net[i]->make_connection(i == 0 ? nullptr : _net[i - 1].get(),
                                 i == (int)_net.size() - 1 ? nullptr : _net[i + 1].get());

      _is_initialized = is_ready();

      return _is_initialized;
    }

    void train(
        tensor2d<T>& train_inputs,
        tensor2d<T>& train_labels,
        std::size_t n_epochs,
        std::size_t n_minibatchs = 1,
        std::function<void(tensor2d<float>&)> initializer = [](tensor2d<float>& x) {
          for (auto& e : x)
            e = (float)0.1;
        }) {
      using namespace fnn::core;
      using namespace fnn::backends;
      context ctx(backend_t::naive, stages::train);

      if (!_is_initialized) {
        // weight initialize function
        auto f = [](tensor2d<float>& x) {
          for (auto& e : x)
            e = (float)0.1;
        };
        initialize(n_minibatchs, f);
      }

      // prepare input and label vectors
      std::vector<tensor2d<T>> train_inputs_batched, train_labels_batched;
      if (train_inputs.template shape<1>() % n_minibatchs) {
        std::cout << "no supported minibatch size" << std::endl;
        return;
      }
      prepare_batched_data(train_inputs, train_labels, train_inputs_batched, train_labels_batched, n_minibatchs);
      std::size_t batchs = train_inputs.size();

      for (std::size_t batch_idx = 0; batch_idx < batchs; batch_idx++) {
        for (int i = 0; i < (int)_net.size(); i++) {
          forward_prop(train_inputs_batched[i], train_labels_batched[i], ctx);
        }
      }
    }

    bool is_ready() {
      bool is_ready = true;
      for (auto& layer : _net)
        is_ready &= layer->is_ready();

      return is_ready;
    }
  };
} // namespace fnn