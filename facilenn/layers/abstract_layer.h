#pragma once

#include "facilenn/backends/backends.h"
#include "facilenn/layers/layer_types.h"
#include "facilenn/utils/utils.h"

namespace fnn {
  namespace layers {
    // forward declaration
    template <typename T>
    class abstract_layer;
    template <typename T>
    class abstract_layer {
      using layer_t = abstract_layer<T>;

     protected:
      // TODO: 基底クラスで定義しないようにする (使わないとき無駄になる && 初期化が面倒)
      tensor2d<T> _in;
      tensor2d<T> _out;
      tensor2d<T> _weight;
      tensor2d<T> _delta;

      std::size_t _in_size;
      std::size_t _out_size;
      std::size_t _n_batch;

      /* DO NOT delete these pointers in destructor! */
      layer_t* _prev_layer;
      layer_t* _next_layer;

      layer_types _layer_type;

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
      virtual tensor2d<T>& weight() { return _weight; }
      virtual tensor2d<T>& delta() { return _delta; }

      virtual tensor2d<T>& forward(tensor2d<T>&, core::context&) = 0;
      virtual tensor2d<T>& backward(tensor2d<T>&, core::context&) = 0;
      virtual tensor2d<T>& optimize(tensor2d<T>&, core::context&) = 0;

      virtual layer_types layer_type() { return _layer_type; }
      virtual void initialize(std::function<void(tensor2d<T>&)>, std::size_t) = 0;
      virtual void make_conenction(layer_t* prev_layer, layer_t* next_layer) {
        _prev_layer = prev_layer;
        _next_layer = next_layer;
      }

      virtual bool is_connected() { return _prev_layer || _next_layer; }
      virtual bool is_connectable() = 0;

      virtual ~abstract_layer() { std::cout << "abst destructor called!" << std::endl; }
    };
  } // namespace layers
} // namespace fnn