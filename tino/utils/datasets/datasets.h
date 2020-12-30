#pragma once

#include "tino/backends/backends.h"
#include "tino/core/context.h"
#include "tino/utils/macros.h"
#include "tino/utils/utils.h"

namespace tino {
  namespace utils {
    template <typename T>
    class dataset_loader {
     private:
      virtual bool check_args(tensor2d<T>&, tensor2d<T>&) = 0;

     public:
      virtual bool generate(tensor2d<T>&, tensor2d<T>&) = 0;

      virtual ~dataset_loader() {}
    };
  } // namespace utils
} // namespace tino