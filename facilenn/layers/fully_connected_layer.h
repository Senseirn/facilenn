#pragma once

#include "facilenn/layers/abstract_layer.h"

namespace fnn {
  namespace layers {
    template <typename T>
    class fully_connected_layer : public abstract_layer<T> {
      using layer_t = abstract_layer<T>;

     private:
      void initialize_weights(std::function<void(tensor2d<T>&)> initializer) {
        initializer(this->_weight);
      }
      void initialize_delta() { std::fill(std::begin(this->_delta), std::end(this->_delta), (T)0); }

     public:
      fully_connected_layer()
      : abstract_layer<T>(layer_types::fully_connected) {}
      fully_connected_layer(std::size_t in_size, std::size_t out_size)
      : abstract_layer<T>(in_size, out_size, layer_types::fully_connected) {}

      tensor2d<T>& forward(tensor2d<T>& prev_out, core::context& ctx) override {
        // temporary return prev_out
        this->_in = prev_out;
        this->_out = this->_in;

        MAYBE_UNUSED(ctx);

        return this->_out;
      }

      tensor2d<T>& backward(tensor2d<T>& next_delta, core::context& ctx) override {
        MAYBE_UNUSED(ctx);

        return next_delta;
      }

      tensor2d<T>& optimize(tensor2d<T>&, core::context& ctx) override {
        MAYBE_UNUSED(ctx);

        return this->_weight;
      }

      void initialize(
          std::function<void(tensor2d<T>&)> initializer =
              [](tensor2d<T>& x) {
                for (auto& e : x)
                  e = (T)0;
              },
          std::size_t n_batch = 1) override {
        this->_n_batch = n_batch;
        this->_in.reshape(this->_n_batch, this->_in_size * this->_out_size);
        this->_weight.reshape(this->_in_size, this->_out_size);
        this->_out.reshape(this->_n_batch, this->_out_size);
        this->_delta.reshape(this->_in_size, this->_out_size);

        initialize_weights(initializer);
        initialize_delta();
      };

      // TODO: implement
      // check if prev layer and next layer are connectable to this layer.
      bool is_connectable() override { return true; }

      ~fully_connected_layer() { std::cout << "fc destructor called!" << std::endl; }
    };
  } // namespace layers
} // namespace fnn