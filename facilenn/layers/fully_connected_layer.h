#pragma once

#include "facilenn/layers/abstract_layer.h"

namespace fnn {
  namespace layers {

    template <typename T>
    class fully_connected_layer : public abstract_layer<T> {
      using layer_t = abstract_layer<T>;

     private:
      std::unique_ptr<abstract_optimizer<T>> _optimizer;

      void initialize_weights(std::function<void(tensor2d<T>&)> initializer) {
        initializer(this->_weight);
      }
      void initialize_delta() { std::fill(std::begin(this->_delta), std::end(this->_delta), (T)0); }

     public:
      fully_connected_layer()
      : abstract_layer<T>(layer_types::fully_connected) {}

      fully_connected_layer(std::size_t in_size, std::size_t out_size)
      : abstract_layer<T>(in_size, out_size, layer_types::fully_connected)
      , _optimizer(nullptr) {}

      fully_connected_layer(std::size_t in_size,
                            std::size_t out_size,
                            std::unique_ptr<abstract_optimizer<T>>&& optimizer)
      : abstract_layer<T>(in_size, out_size, layer_types::fully_connected)
      , _optimizer(std::move(optimizer)) {}

      tensor2d<T>& forward(tensor2d<T>& prev_out, core::context& ctx) override {

        // temporary return prev_out
        this->_in = prev_out;
        this->_out = this->_in;

        FNN_MAYBE_UNUSED(ctx);

        if (!this->_next_layer)
          this->_next_layer->forward(op::fully_connected_forward_kernel(
                                         this->_in, this->_weight, this->_bias, this->_out, ctx),
                                     ctx);

        return this->_out;
      }

      tensor2d<T>& backward(tensor2d<T>& next_delta, core::context& ctx) override {
        FNN_MAYBE_UNUSED(next_delta);
        FNN_MAYBE_UNUSED(ctx);

        if (!this->_prev_layer)
          this->_prev_layer->backward(this->_delta, ctx);

        return this->_delta;
      }

      tensor2d<T>& optimize(tensor2d<T>& next_delta, core::context& ctx) override {
        FNN_MAYBE_UNUSED(next_delta);
        FNN_MAYBE_UNUSED(ctx);

        return this->_weight;
      }

      bool initialize(
          std::function<void(tensor2d<T>&)> initializer =
              [](tensor2d<T>& x) {
                for (auto& e : x)
                  e = (T)0;
              },
          std::size_t n_batch = 1) override {
        if (!check_connection())
          return false;

        this->_n_batch = n_batch;
        this->_in.reshape(this->_n_batch, this->_in_size);
        this->_weight.reshape(this->_in_size, this->_out_size);
        this->_bias.reshape(1, this->_out_size);
        this->_out.reshape(this->_n_batch, this->_out_size);
        this->_delta.reshape(this->_in_size, this->_out_size);

        initialize_weights(initializer);
        initialize_delta();

        return true;
      };

      void set_optimizer(std::unique_ptr<abstract_optimizer<T>> optimizer) override {
        this->_optimizer = std::move(optimizer);
      }

      // TODO: implement
      // check if prev layer and next layer are connectable to this layer.
      bool check_connection() override {
        if (!this->is_connected())
          return false;

        if (this->_prev_layer &&
            this->_prev_layer->out().num_elements() != this->_in.num_elements())
          return false;

        if (this->_next_layer &&
            this->_next_layer->in().num_elements() != this->_out.num_elements())
          return false;

        return true;
      }

      ~fully_connected_layer() {}
    };
  } // namespace layers
} // namespace fnn