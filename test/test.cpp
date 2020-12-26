#include "../tino/tino.h"

int main() {
  using namespace tino;
  using namespace tino::layers;
  using namespace tino::core;
  using namespace tino::utils;

  // define network
  tino::network net;
  net.add(new fully_connected_layer(64, 32));
  net.add(new relu_layer());
  net.add(new fully_connected_layer(32, 16));
  net.add(new relu_layer());
  net.add(new fully_connected_layer(16, 1));
  net.add(new relu_layer());
  //  net.initialize();

  // define how to initialize weights
  net.weight_initializer([](tensor2d<float>& x) {
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<> rand(-0.5, 0.5);
    for (auto& e : x)
      e = rand(mt);
  });

  // declare optimizer
  // here we use SGD with paramert alpha = 0.2
  optimizers::SGD sgd;
  sgd.alpha(0.2);

  tensor2d<float> train_inputs(1000, 2);
  tensor2d<float> train_labels(1000, 1);
  xor_generator generator;
  generator.generate(train_inputs, train_labels);

  int n_epochs = 10;
  int n_minibatchs = 10;

  // run train
  net.train(train_inputs, train_labels, n_epochs, n_minibatchs, sgd);
}