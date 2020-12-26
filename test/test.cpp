#include "../tino/tino.h"

int main() {
  using namespace tino;
  using namespace tino::layers;
  using namespace tino::core;
  using namespace tino::utils;

  tino::network<float> net;
  net.add(new fully_connected_layer<float>(10, 10, sgd<float>(0.1)));
  net.add(new relu_layer<float>());
  net.add(new fully_connected_layer<float>(10, 1, sgd<float>(0.1)));
  net.add(new relu_layer<float>());
  //  net.initialize();

  tensor2d<float> train_inputs(1000, 2);
  tensor2d<float> train_labels(1000, 1);
  xor_generator<float> generator;
  generator.generate(train_inputs, train_labels);

  net.train(train_inputs, train_labels, 1, 1, [](tensor2d<float>& x) {
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<> rand(-0.5, 0.5);
    for (auto& e : x)
      e = rand(mt);
  });

  std::cout << "is ready: " << net.is_ready() << std::endl;
}