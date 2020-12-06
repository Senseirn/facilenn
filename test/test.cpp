#include "../facilenn/facilenn.h"

int main() {
  using namespace fnn;
  using namespace fnn::layers;
  using namespace core::optimizers;

  fnn::network<float> net;
  net.add(new fully_connected_layer<float>(2, 4, sgd<float>(0.1)));
  net.add(new relu_layer<float>());
  net.add(new fully_connected_layer<float>(4, 1));
  net.add(new softmax_layer<float>());

  net.initialize();

  std::cout << net.is_ready() << std::endl;
}