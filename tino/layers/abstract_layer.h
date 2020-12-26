#pragma once

#include "tino/backends/backends.h"
#include "tino/layers/layer_types.h"
#include "tino/utils/utils.h"

namespace tino {
  namespace layers {
    using namespace core;
    using namespace optimizers;
    // forward declaration
    template <typename T>
    class abstract_layer;
    template <typename T>
    class abstract_layer {
      using layer_t = abstract_layer<T>;

     protected:
      // TODO: can I make it better here?
      tensor2d<T> _in;
      tensor2d<T> _out;
      tensor2d<T> _bias;
      tensor2d<T> _weight;
      tensor2d<T> _delta;
      tensor2d<T> _delta_weight;
      tensor2d<T> _delta_bias;

      std::size_t _in_size;
      std::size_t _out_size;
      std::size_t _n_batch;

      /* DO NOT delete these pointers in destructor! */
      layer_t* _prev_layer = nullptr;
      layer_t* _next_layer = nullptr;

      layer_types _layer_type;

     protected:
      virtual bool is_connected() { return _prev_layer || _next_layer; }

      virtual bool check_connection() {
        if (!this->is_connected())
          return false;

        if (this->_prev_layer && this->_prev_layer->out().num_elements() != this->_in.num_elements())
          return false;

        if (this->_next_layer && this->_next_layer->in().num_elements() != this->_out.num_elements())
          return false;

        return true;
      }

     public:
      abstract_layer(layer_types layer_type)
      : _layer_type(layer_type) {}
      abstract_layer(std::size_t in_size, std::size_t out_size, layer_types layer_type)
      : _in_size(in_size)
      , _out_size(out_size)
      , _n_batch(1)
      , _layer_type(layer_type) {}

      virtual tensor2d<T>& in() { return _in; }
      virtual tensor2d<T>& out() { return _out; }
      virtual tensor2d<T>& bias() { return _bias; }
      virtual tensor2d<T>& weight() { return _weight; }
      virtual tensor2d<T>& delta() { return _delta; }
      virtual tensor2d<T>& delta_weight() { return _delta_weight; }
      virtual tensor2d<T>& delta_bias() { return _delta_bias; }

      virtual tensor2d<T>& forward(tensor2d<T>&, core::context&) = 0;
      virtual tensor2d<T>& backward(tensor2d<T>&, core::context&) = 0;
      virtual tensor2d<T>& optimize(tensor2d<T>&, core::context&) = 0;

      virtual layer_types layer_type() { return _layer_type; }
      virtual bool initialize(std::function<void(tensor2d<T>&)>, std::size_t) = 0;

      virtual void set_optimizer(std::unique_ptr<abstract_optimizer_<T>>) {}

      virtual void make_connection(layer_t* prev_layer, layer_t* next_layer) {
        _prev_layer = prev_layer;
        _next_layer = next_layer;
      }

      virtual bool is_ready() { return this->check_connection(); }

      virtual ~abstract_layer() {}
    };
  } // namespace layers
} // namespace tino