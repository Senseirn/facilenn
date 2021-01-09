#pragma once

#include "backends/backends.h"
#include "core/core.h"
#include "layers/layers.h"
#include "loss/loss.h"
#include "models/models.h"
#include "utils/utils.h"

#include <chrono>

namespace tino {
  template <typename T>
  class netowork_;

  template <typename T = TINO_FLOAT_TYPE>
  class network_ {
   private:
    std::vector<std::unique_ptr<layers::abstract_layer<T>>> _net;
    std::function<void(tensor2d<T>&)> _weight_initializer;
    bool _is_initialized                             = false;
    core::optimizers::optimizer_t _current_optimizer = core::optimizers::optimizer_t::none;

    void prepare_batched_data(tensor2d<T>& inputs, tensor2d<T>& labels, std::vector<tensor2d<T>>& batched_inputs, std::vector<tensor2d<T>>& batched_labels, std::size_t n_minibatchs) {
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

    // TODO: use template specilization instead of swich-case
    template <loss::loss_t loss_func>
    T calc_loss(tensor2d<T>& label, core::context& ctx) {
      switch (loss_func) {
        case loss::loss_t::mse: return loss::mse_<T>::f(_net.back()->out(), label, ctx); break;
        case loss::loss_t::cross_entropy: return loss::cross_entropy_<T>::f(_net.back()->out(), label, ctx); break;
        default: break;
      }
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
        _net[i]->make_connection(i == 0 ? nullptr : _net[i - 1].get(), i == (int)_net.size() - 1 ? nullptr : _net[i + 1].get());

      for (auto& l : _net) {
        l->initialize(initializer, n_batch);
      }
      _is_initialized = is_ready();

      return _is_initialized;
    }

    template <loss::loss_t loss_func, class Optimizer>
    void train(tensor2d<T>& train_inputs, tensor2d<T>& train_labels, std::size_t n_epochs, std::size_t n_batchsize, const Optimizer& optimizer, core::context& ctx) {
      using namespace tino::core;
      using namespace tino::backends;
      ctx.stage(stages::train);

      if (!_is_initialized) {
        // weight initialize function
        initialize(_weight_initializer, n_batchsize);
        set_optimizer_each_layer(optimizer);
      }

      // set optimizer for each layer
      if (_current_optimizer != optimizer.type()) {
        set_optimizer_each_layer(optimizer);
        _current_optimizer = optimizer.type();
      }

      // prepare input and label vectors
      std::vector<tensor2d<T>> train_inputs_batched, train_labels_batched;
      if (train_inputs.template shape<1>() % n_batchsize) {
        std::cout << "no supported minibatch size" << std::endl;
        return;
      }
      prepare_batched_data(train_inputs, train_labels, train_inputs_batched, train_labels_batched, n_batchsize);
      std::size_t n_minibatchs = train_inputs.template shape<1>() / n_batchsize;
      std::size_t n_samples    = n_minibatchs * train_inputs_batched[0].template shape<1>();

      std::string output_text;
      std::chrono::system_clock::time_point start = std::chrono::system_clock::now(), point = std::chrono::system_clock::now();
      for (std::size_t epoch = 1; epoch <= n_epochs; epoch++) {
        std::cout << "epoch " << epoch << std::endl;
        T loss            = 0;
        int correct_count = 0;
        output_text       = "- ";
        for (std::size_t batch_idx = 0; batch_idx < n_minibatchs; batch_idx++) {
          forward(train_inputs_batched[batch_idx], ctx);
          loss += calc_loss<loss_func>(train_labels_batched[batch_idx], ctx);
          backward(train_labels_batched[batch_idx], ctx);

          if (1)
            for (int b = 0; b < (int)n_batchsize; b++) {
              int idx     = -1;
              int max_idx = -1;
              T maximum   = std::numeric_limits<T>::lowest();
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

        output_text += "loss: " + std::to_string(loss / n_samples) + " - ";
        output_text += "acc: " + std::to_string((float)correct_count / n_samples * 100) + "% - ";

        float elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count() / 1000.f;
        output_text += "elaplsed: " + std::to_string(elapsed) + "s - ";

        float duration   = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - point).count() / 1000.f;
        float per_sample = duration / n_samples;
        output_text += "duration: " + std::to_string(duration) + "s (=" + std::to_string(per_sample * 1000 * 1000) + "us/sample)";
        std::cout << output_text << std::endl;

        point = std::chrono::system_clock::now();
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