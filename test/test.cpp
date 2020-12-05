#include "../facilenn/facilenn.h"

int main() {
  using namespace fnn;
  using namespace fnn::layers;
  using namespace core::optimizers;

  auto fc1 = new fully_connected_layer<float>(2, 4);
  auto fc2 = new fully_connected_layer<float>(4, 3);

  auto weight_initiaizer = [](tensor2d<float>& x) {
    for (auto& e : x)
      e = (float)0.1;
  };

  fc1->initialize(weight_initiaizer, 5);
  fc2->initialize(weight_initiaizer, 5);

  fc1->make_connection(nullptr, fc2);
  fc2->make_connection(fc1, nullptr);

  if (fc1->check_connection())
    std::cout << "fc1 connectable!" << std::endl;

  if (fc1->check_connection())
    std::cout << "connected!" << std::endl;

  for (auto e : fc1->weight()) {
    std::cout << e << std::endl;
  }

  std::cout << (fc1->layer_type() == layer_types::fully_connected) << std::endl;

  std::cout << "build success" << std::endl;

  fnn::network<float> net;
  net.add(new fully_connected_layer<float>(2, 4, sgd<float>(0.1)));
  net.add(new relu_layer<float>());
  net.add(new fully_connected_layer<float>(4, 1));
  net.add(new softmax_layer<float>());
  net.initialize();
}