#pragma once

#include "tino/core/op/activation_softmax_ops.h"
#include "tino/layers/abstract_layer.h"

namespace tino {
  namespace layers {
    template <typename T = TINO_FLOAT_TYPE>
    class softmax_layer_ : public abstract_layer<T> {
      using layer_t = abstract_layer<T>;

     private:
     public:
      softmax_layer_()
      : abstract_layer<T>(layer_types::softmax) {}

      softmax_layer_(std::size_t in_size, std::size_t out_size)
      : abstract_layer<T>(in_size, out_size, layer_types::softmax) {}

      tensor2d<T>& forward(tensor2d<T>& prev_out, core::context& ctx) override {

        // temporary return prev_out
        this->_in = prev_out;

        if (this->_next_layer)
          this->_next_layer->forward(op::softmax_activation_forward_kernel(this->_in, this->_out, ctx), ctx);
        else
          op::softmax_activation_forward_kernel(this->_in, this->_out, ctx);

        return this->_out;
      }

      tensor2d<T>& backward(tensor2d<T>& next_delta, core::context& ctx) override {
        TINO_MAYBE_UNUSED(next_delta);
        TINO_MAYBE_UNUSED(ctx);

        if (!this->_next_layer) {
          using index_t = typename tensor2d<T>::index_t;
          for (index_t i = 0; i < next_delta.template shape<1>(); i++)
            for (index_t j = 0; j < next_delta.template shape<0>(); j++)
              this->_delta(i, j) = this->_out(i, j) - next_delta(i, j);
        }
        if (this->_prev_layer) {
          if (!this->_next_layer)
            this->_prev_layer->backward(this->_delta, ctx);
          else
            this->_prev_layer->backward(
                op::softmax_activation_backward_kernel(this->_in, this->_delta, next_delta, ctx), ctx);
        }
        if (!this->_next_layer)
          optimize(next_delta, ctx);
        return this->_delta;
      }

      tensor2d<T>& optimize(tensor2d<T>& next_delta, core::context& ctx) override {
        TINO_MAYBE_UNUSED(next_delta);
        TINO_MAYBE_UNUSED(ctx);

        if (this->_prev_layer)
          this->_prev_layer->optimize(this->_delta, ctx);

        return this->_weight;
      }

      bool initialize(
          std::function<void(tensor2d<T>&)> initializer =
              [](tensor2d<T>& x) {
                for (auto& e : x)
                  e = (T)0;
              },
          std::size_t n_batch = 1) override {
        if (!this->is_connected())
          return false;

        auto& out = this->_prev_layer->out();

        this->_in_size = out.template shape<0>();
        this->_out_size = out.template shape<0>();
        this->_n_batch = n_batch;
        this->_in.reshape(this->_n_batch, this->_in_size);
        this->_out.reshape(this->_n_batch, this->_out_size);
        this->_delta.reshape(this->_n_batch, this->_out_size);

        TINO_MAYBE_UNUSED(initializer);

        return true;
      };

      bool is_optimizer_set() override { return false; }

      ~softmax_layer_() {}
    };
    using softmax_layer = softmax_layer_<TINO_FLOAT_TYPE>;
  } // namespace layers
} // namespace tino