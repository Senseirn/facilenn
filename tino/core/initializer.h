#pragma once

#include "tino/backends/backends.h"
#include "tino/core/context.h"
#include "tino/utils/utils.h"

namespace tino {
  namespace core {
    namespace initializers {
      template <typename T = TINO_FLOAT_TYPE>
      void uniform_rand(tensor2d<T>& weights) {
        std::random_device rnd;
        std::mt19937 mt(rnd());
        std::uniform_real_distribution<> rand(-0.5, 0.5);
        for (auto& e : weights)
          e = rand(mt);
      }

      template <typename T = TINO_FLOAT_TYPE>
      void Xavier(tensor2d<T>& weights) {
        std::random_device rnd;
        std::mt19937 mt(rnd());
        std::normal_distribution<> rand(0.f, 0.1f);
        const T sn = std::sqrt(weights.template shape<1>());
        for (auto& e : weights)
          e = rand(mt) / sn;
      }

      template <typename T = TINO_FLOAT_TYPE>
      void He(tensor2d<T>& weights) {
        std::random_device rnd;
        std::mt19937 mt(rnd());
        std::normal_distribution<> rand(0.f, 0.1f);
        const T sn = std::sqrt(2.f / weights.template shape<1>());
        for (auto& e : weights) {
          e = rand(mt) / sn;
        }
      }
    } // namespace initializers
  }   // namespace core
} // namespace tino