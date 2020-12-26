#pragma once

#include "tino/backends/backends.h"
#include "tino/core/context.h"
#include "tino/utils/macros.h"
#include "tino/utils/utils.h"

#include <random>

namespace tino {
  namespace utils {
    template <typename T>
    class dataset_generator {
     private:
      virtual bool check_args(tensor2d<T>&, tensor2d<T>&) = 0;

     public:
      virtual bool generate(tensor2d<T>&, tensor2d<T>&) = 0;

      virtual ~dataset_generator() {}
    };

    template <typename T = TINO_FLOAT_TYPE>
    class xor_generator_ : public dataset_generator<T> {
     private:
      std::size_t _n_data;
      tensor2d<T> _train_inputs;
      tensor2d<T> _train_labels;

      bool check_args(tensor2d<T>& train_inputs, tensor2d<T>& train_labels) {
        if (train_inputs.template shape<1>() <= 0 || train_labels.template shape<1>() <= 0)
          return false;
        if (train_inputs.template shape<1>() != train_labels.template shape<1>())
          return false;
        if (train_inputs.template shape<0>() != 2)
          return false;
        if (train_labels.template shape<0>() != 1)
          return false;

        return true;
      }

     public:
      xor_generator_(const std::size_t n_data)
      : _n_data(n_data)
      , _train_inputs(_n_data, 2)
      , _train_labels(_n_data, 1) {
        generate(_train_inputs, _train_labels);
      }

      bool generate(tensor2d<T>& train_inputs, tensor2d<T>& train_labels) {
        if (!check_args(train_inputs, train_labels))
          return false;

        std::random_device rnd;
        std::mt19937 mt(rnd());
        std::uniform_int_distribution<> dist(0, 1);

        for (std::size_t i = 0; i < train_inputs.template shape<1>(); i++) {
          const auto x1 = dist(mt);
          const auto x2 = dist(mt);
          const auto l1 = x1 == x2 ? 0 : 1;
          train_inputs(i, 0) = x1;
          train_inputs(i, 1) = x2;
          train_labels(i, 0) = l1;
        }
        return true;
      }

      tensor2d<T>& train_inputs() { return _train_inputs; }
      tensor2d<T>& train_labels() { return _train_labels; }

      ~xor_generator_() {}
    };
    using xor_generator = xor_generator_<TINO_FLOAT_TYPE>;
  } // namespace utils
} // namespace tino