#include "../tino/tino.h"

int main() {
  using namespace tino;
  using namespace tino::layers;
  using namespace tino::core;
  using namespace tino::utils;

  tino::network net;
  SGD sgd;
  sgd.alpha(0.2);
  net.add(new fully_connected_layer(64, 32, sgd()));
  net.add(new relu_layer());
  net.add(new fully_connected_layer(32, 16, sgd()));
  net.add(new relu_layer());
  net.add(new fully_connected_layer(16, 1, sgd()));
  net.add(new relu_layer());
  //  net.initialize();

  tensor2d<float> train_inputs(1000, 2);
  tensor2d<float> train_labels(1000, 1);
  xor_generator generator;
  generator.generate(train_inputs, train_labels);

  net.train(train_inputs, train_labels, 30, 20, [](tensor2d<float>& x) {
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<> rand(-0.5, 0.5);
    for (auto& e : x)
      e = rand(mt);
  });
}