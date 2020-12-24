#include "../tino/tino.h"

int main() {
  using namespace tino;
  using namespace tino::layers;
  using namespace tino::core;
  using namespace tino::utils;

  tino::network<float> net;
  net.add(new fully_connected_layer<float>(2, 2, sgd<float>(0.1)));
  net.add(new relu_layer<float>());
  net.add(new fully_connected_layer<float>(2, 1));
  net.add(new relu_layer<float>());
  //  net.initialize();

  tensor2d<float> train_inputs(10, 2);
  tensor2d<float> train_labels(10, 1);
  xor_generator<float> generator;
  generator.generate(train_inputs, train_labels);

  net.train(train_inputs, train_labels, 1, 5);

  std::cout << "is ready: " << net.is_ready() << std::endl;
}