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

  // run 10 epochs with batch_size=10
  int n_epochs = 10;
  int n_batchsize = 10;

  // generate xor dataset which contains 1,000 pairs of input and label
  xor_generator generator(1000);

  // run train
  net.train(generator.train_inputs(), generator.train_labels(), n_epochs, n_batchsize, sgd);
}