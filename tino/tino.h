#pragma once

#include "backends/backends.h"
#include "core/core.h"
#include "layers/layers.h"
#include "loss/loss.h"
#include "models/models.h"
#include "utils/utils.h"

namespace tino {
  template <typename T>
  class netowork_;

  template <typename T = TINO_FLOAT_TYPE>
  class network_ {
   private:
    std::vector<std::unique_ptr<layers::abstract_layer<T>>> _net;
    std::function<void(tensor2d<T>&)> _weight_initializer;
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

    template <class Optimizer>
    void set_optimizer_each_layer(const Optimizer& optimizer) {
      for (auto& layer : _net)
        layer->set_optimizer(optimizer());
    }

    T calc_loss(tensor2d<T>& label, core::context& ctx) {
      // return loss::mse<T>::f(_net.back()->out(), label, ctx);
      return loss::cross_entropy<T>::f(_net.back()->out(), label, ctx);
    }

    void forward(tensor2d<T>& input, core::context& ctx) { _net.front()->forward(input, ctx); }

    void backward(tensor2d<T>& label, core::context& ctx) { _net.back()->backward(label, ctx); }

   public:
    void add(layers::abstract_layer<T>* layer) { _net.emplace_back(layer); }

    network_& weight_initializer(std::function<void(tensor2d<T>&)> func) {
      _weight_initializer = func;
      return *this;
    }

    template <typename F>
    bool initialize(F initializer, std::size_t n_batch = 1) {
      for (int i = 0; i < (int)_net.size(); i++)
        _net[i]->make_connection(i == 0 ? nullptr : _net[i - 1].get(),
                                 i == (int)_net.size() - 1 ? nullptr : _net[i + 1].get());

      for (auto& l : _net) {
        l->initialize(initializer, n_batch);
      }
      _is_initialized = is_ready();

      return _is_initialized;
    }

    template <class Optimizer>
    void train(tensor2d<T>& train_inputs,
               tensor2d<T>& train_labels,
               std::size_t n_epochs,
               std::size_t n_batchsize,
               const Optimizer& optimizer) {
      using namespace tino::core;
      using namespace tino::backends;
      context ctx(backend_t::naive, stages::train);

      if (!_is_initialized) {
        // weight initialize function
        initialize(_weight_initializer, n_batchsize);
      }

      // set optimizer for each layer
      set_optimizer_each_layer(optimizer);

      // prepare input and label vectors
      std::vector<tensor2d<T>> train_inputs_batched, train_labels_batched;
      if (train_inputs.template shape<1>() % n_batchsize) {
        std::cout << "no supported minibatch size" << std::endl;
        return;
      }
      prepare_batched_data(train_inputs, train_labels, train_inputs_batched, train_labels_batched, n_batchsize);
      std::size_t n_minibatchs = train_inputs.template shape<1>() / n_batchsize;

      for (std::size_t epoch = 1; epoch <= n_epochs; epoch++) {
        std::cout << "epoch: " << epoch << std::endl;
        T loss = 0;
        T accuracy = 0;
        int correct_count = 0;
        for (std::size_t batch_idx = 0; batch_idx < n_minibatchs; batch_idx++) {
          forward(train_inputs_batched[batch_idx], ctx);
          loss += calc_loss(train_labels_batched[batch_idx], ctx);
          backward(train_labels_batched[batch_idx], ctx);

          /*
                    std::cout << batch_idx << ": input: " << train_inputs_batched[batch_idx](0, 0) << " "
                              << train_inputs_batched[batch_idx](0, 1) << " " << train_labels_batched[batch_idx](0, 0)
             << " "
                              << _net.back()->out()(0, 0) << " " << std::endl;
                              */

          for (int b = 0; b < n_batchsize; b++) {
            int idx = -1;
            int max_idx = -1;
            T maximum = std::numeric_limits<T>::lowest();
            for (int i = 0; i < 10; i++) {
              if (train_labels_batched[batch_idx](b, i) > 0.99) {
                idx = i;
              }
            }
            for (int i = 0; i < 10; i++) {
              if (_net.back()->out()(b, i) > maximum) {
                max_idx = i;
                maximum = _net.back()->out()(b, i);
              }
            }
            if (idx == max_idx) {
              correct_count++;
            }
          }
        }
        std::cout << "loss: " << loss / (n_minibatchs * train_inputs_batched[0].template shape<1>()) << std::endl;
        std::cout << "acc: "
                  << (float)correct_count / (n_minibatchs * train_inputs_batched[0].template shape<1>()) * 100 << " %"
                  << std::endl;
      }
    }

    bool is_ready() {
      bool is_ready = true;
      for (auto& layer : _net)
        is_ready &= layer->is_ready();

      return is_ready;
    }

    ~network_() {}
  };
  using network = network_<TINO_FLOAT_TYPE>;
} // namespace tino