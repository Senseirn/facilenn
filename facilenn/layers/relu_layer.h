#pragma once

#include "facilenn/core/op/activation_relu_ops.h"
#include "facilenn/layers/abstract_layer.h"

namespace fnn {
  namespace layers {
    template <typename T>
    class relu_layer : public abstract_layer<T> {
      using layer_t = abstract_layer<T>;

     private:
      /*
      std::unique_ptr<abstract_optimizer<T>> _optimizer;

      void initialize_weights(std::function<void(tensor2d<T>&)> initializer) {
        initializer(this->_weight);
      }
      void initialize_delta() { std::fill(std::begin(this->_delta), std::end(this->_delta), (T)0); }
      */

     public:
      relu_layer()
      : abstract_layer<T>(layer_types::relu) {}

      relu_layer(std::size_t in_size, std::size_t out_size)
      : abstract_layer<T>(in_size, out_size, layer_types::relu) {}

      tensor2d<T>& forward(tensor2d<T>& prev_out, core::context& ctx) override {

        // temporary return prev_out
        this->_in = std::move(prev_out);

        FNN_MAYBE_UNUSED(ctx);

        if (!this->_next_layer)
          this->_next_layer->forward(op::relu_activation_forward_kernel(this->_in, this->_out, ctx), ctx);

        return this->_out;
      }

      tensor2d<T>& backward(tensor2d<T>& next_delta, core::context& ctx) override {

        if (!this->_prev_layer)
          this->_prev_layer->backward(op::relu_activation_backward_kernel(this->_in, this->_delta, next_delta, ctx),
                                      ctx);

        return this->_delta;
      }

      tensor2d<T>& optimize(tensor2d<T>& next_delta, core::context& ctx) override {
        FNN_MAYBE_UNUSED(next_delta);
        FNN_MAYBE_UNUSED(ctx);

        return this->_delta;
      }

      bool initialize(
          std::function<void(tensor2d<T>&)> initializer =
              [](tensor2d<T>& x) {
                for (auto& e : x)
                  e = (T)0;
              },
          std::size_t n_batch = 1) override {
        if (!this->check_connection())
          return false;

        auto& out = this->_prev_layer->out();

        this->_in_size = out.template shape<0>();
        this->_out_size = out.template shape<0>();
        this->_n_batch = n_batch;
        this->_in.reshape(this->_n_batch, this->_in_size);
        this->_out.reshape(this->_n_batch, this->_out_size);

        FNN_MAYBE_UNUSED(initializer);

        return true;
      };

      /*
            void set_optimizer(std::unique_ptr<abstract_optimizer<T>> optimizer) override {
              this->_optimizer = std::move(optimizer);
            }
      */

      ~relu_layer() {}
    };
  } // namespace layers
} // namespace fnn