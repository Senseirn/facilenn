#include "../tino/tino.h"

int main() {
  using namespace tino;
  using namespace tino::layers;
  using namespace tino::core;

  tino::network<float> net;
  net.add(new fully_connected_layer<float>(2, 4, sgd<float>(0.1)));
  net.add(new relu_layer<float>());
  net.add(new fully_connected_layer<float>(4, 1));
  net.add(new softmax_layer<float>());

  net.initialize();

  std::cout << net.is_ready() << std::endl;
}