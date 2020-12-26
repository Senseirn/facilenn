#pragma once

#include "backends/backends.h"
#include "core/core.h"
#include "layers/layers.h"
#include "loss/loss.h"
#include "models/models.h"
#include "utils/utils.h"

namespace tino {
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
        batched_inputs[i].reshape(n_minibatchs, inputs.template shape<0>());
        batched_labels[i].reshape(n_minibatchs, labels.template shape<0>());
        for (std::size_t j = 0; j < n_minibatchs; j++) {
          for (std::size_t k = 0; k < inputs.template shape<0>(); k++)
            batched_inputs[i](j, k) = inputs(i * n_minibatchs + j, k);
          for (std::size_t k = 0; k < labels.template shape<0>(); k++)
            batched_labels[i](j, k) = labels(i * n_minibatchs + j, k);
        }
      }
    }

    T forward_prop(tensor2d<T>& input, tensor2d<T>& label, core::context& ctx) {
      _net.front()->forward(input, ctx);
      return loss::mse<T>::f(_net.back()->out(), label, ctx);
    }

    void backward_prop(tensor2d<T>& label, core::context& ctx) {
      _net.back()->backward(label, ctx);
      // return loss::mse<T>::f(_net.back()->out(), label, ctx);
    }

   public:
    void add(layers::abstract_layer<T>* layer) { _net.emplace_back(layer); }

    bool initialize(
        std::size_t n_batch = 1,
        std::function<void(tensor2d<float>&)> initializer = [](tensor2d<float>& x) {
          for (auto& e : x)
            e = (float)0.1;
        }) {

      for (int i = 0; i < (int)_net.size(); i++)
        _net[i]->make_connection(i == 0 ? nullptr : _net[i - 1].get(),
                                 i == (int)_net.size() - 1 ? nullptr : _net[i + 1].get());

      for (auto& l : _net) {
        l->initialize(initializer, n_batch);
      }
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
            e = (float)0.01;
        }) {
      using namespace tino::core;
      using namespace tino::backends;
      context ctx(backend_t::naive, stages::train);

      if (!_is_initialized) {
        // weight initialize function
        initialize(n_minibatchs, initializer);
      }

      // prepare input and label vectors
      std::vector<tensor2d<T>> train_inputs_batched, train_labels_batched;
      if (train_inputs.template shape<1>() % n_minibatchs) {
        std::cout << "no supported minibatch size" << std::endl;
        return;
      }
      prepare_batched_data(train_inputs, train_labels, train_inputs_batched, train_labels_batched, n_minibatchs);
      std::size_t batchs = train_inputs.template shape<1>() / n_minibatchs;

      for (std::size_t epoch = 1; epoch <= n_epochs; epoch++) {
        std::cout << "epoch: " << epoch << std::endl;
        for (std::size_t batch_idx = 0; batch_idx < batchs; batch_idx++) {

          auto loss = forward_prop(train_inputs_batched[batch_idx], train_labels_batched[batch_idx], ctx);
          backward_prop(train_labels_batched[batch_idx], ctx);

          std::cout << batch_idx << ": input: " << train_inputs_batched[batch_idx](0, 0) << " "
                    << train_inputs_batched[batch_idx](0, 1) << " " << train_labels_batched[batch_idx](0, 0) << " "
                    << _net.back()->out()(0, 0) << " " << loss << std::endl;
        }
      }
    }

    bool is_ready() {
      bool is_ready = true;
      for (auto& layer : _net)
        is_ready &= layer->is_ready();

      return is_ready;
    }

    ~network() {}
  };
} // namespace tino