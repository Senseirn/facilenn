#pragma once

#include "tino/backends/backends.h"
#include "tino/core/context.h"
#include "tino/utils/datasets/datasets.h"
#include "tino/utils/macros.h"
#include "tino/utils/utils.h"

#include <fstream>
#include <string>

namespace tino {
  namespace utils {
    template <typename T = TINO_FLOAT_TYPE>
    class cifar10_loader_ : public dataset_loader<T> {
     private:
      const std::size_t _n_train_data_per_bin = 10000;
      const std::size_t _n_train_batch_bin    = 5;
      const std::size_t _n_total_train_data   = _n_train_data_per_bin * _n_train_batch_bin;
      const std::size_t _n_test_data          = 10000;
      const std::size_t _image_width          = 32;
      const std::size_t _image_height         = 32;
      const std::size_t _image_ch             = 3;
      const std::size_t _n_classes            = 10;

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

      int to_little_endian(int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;

        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
      }

      // NCHW
      bool read_images(const std::string train_image_dir_path) {

        for (int bin_idx = 0; bin_idx < (int)_n_train_batch_bin; bin_idx++) {
          const std::string filename = train_image_dir_path + "/data_batch_" + std::to_string(bin_idx + 1) + ".bin";
          std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
          if (ifs.fail()) {
            std::cerr << "failed to open file" << std::endl;
            std::exit(1);
          }

          // read pixel data
          unsigned char c     = 0;
          unsigned char label = -1;
          for (std::size_t n = 0; n < _n_train_data_per_bin; n++) {
            ifs.read(reinterpret_cast<char*>(&label), sizeof(unsigned char));
            int img_idx                       = n + bin_idx * _n_train_data_per_bin;
            int label_idx                     = label;
            _train_labels(img_idx, label_idx) = (T)1;
            for (std::size_t ch = 0; ch < _image_ch; ch++) {
              for (std::size_t h = 0; h < _image_height; h++) {
                for (std::size_t w = 0; w < _image_width; w++) {
                  ifs.read(reinterpret_cast<char*>(&c), sizeof(unsigned char));
                  _train_inputs(img_idx, ch * _image_height * _image_width + h * _image_width + w) = ((T)c / 255.f);
                  c                                                                                = 0;
                }
              }
            }
          }
        }

        return true;
      }

      bool read_labels(const std::string train_label_path) {
        /*
              constexpr int magic_number = 8;

              std::ifstream ifs(train_label_path.c_str(), std::ios::in | std::ios::binary);
              if (ifs.fail()) {
                std::cerr << "failed to open file" << std::endl;
                std::exit(1);
              }

              // skip first 16 bytes
              ifs.ignore(magic_number);

              // read pixel data
              unsigned char c = 0;
              for (std::size_t i = 0; i < _n_train_data; i++) {
                ifs.read(reinterpret_cast<char*>(&c), sizeof(unsigned char));
                int idx = c;
                _train_labels(i, idx) = (T)1;
                c = 0;
              }
      */
        return true;
      }

     public:
      cifar10_loader_(std::string train_image_dir_path)
      : _train_inputs(_n_total_train_data, _image_ch * _image_width * _image_height)
      , _train_labels(_n_total_train_data, _n_classes) {
        _train_inputs.fill(0);
        _train_labels.fill(0);
        read_images(train_image_dir_path);
        //     read_labels(train_label_path);
      }

      bool generate(tensor2d<T>& train_inputs, tensor2d<T>& train_labels) {
        TINO_MAYBE_UNUSED(train_inputs);
        TINO_MAYBE_UNUSED(train_labels);
        return true;
      }

      tensor2d<T>& train_inputs() { return this->_train_inputs; }
      tensor2d<T>& train_labels() { return _train_labels; }

      ~cifar10_loader_() {}
    }; // namespace utils
    using cifar10_loader = cifar10_loader_<TINO_FLOAT_TYPE>;
  } // namespace utils
} // namespace tino