#include "../tino/tino.h"

int main() {
  using namespace tino;
  using namespace tino::layers;
  using namespace tino::core;
  using namespace tino::utils;
  using namespace tino::loss;

  // define network
  tino::network net;
  net.add(new fully_connected_layer(28 * 28, 64 * 3));
  net.add(new relu_layer());
  net.add(new fully_connected_layer(64 * 3, 32 * 4));
  net.add(new relu_layer());
  net.add(new fully_connected_layer(32 * 4, 10));
  net.add(new softmax_layer());

  // define how to initialize weights
  /*
    net.weight_initializer([](tensor2d<float>& x) {
      std::random_device rnd;
      std::mt19937 mt(rnd());
      std::uniform_real_distribution<> rand(-0.5, 0.5);
      for (auto& e : x)
        e = rand(mt);
    });
  */
  // here we use He initializer
  net.weight_initializer(initializers::He<TINO_FLOAT_TYPE>);

  // declare optimizer
  // here we use SGD with parameter alpha = 0.2
  optimizers::SGD sgd;
  sgd.alpha(0.01f);

  optimizers::Adam adam;
  adam.alpha(0.001f);

  // run 10 epochs with batch_size=10
  int n_epochs = 20;
  int n_batchsize = 200;

  // generate xor dataset which contains 1,000 pairs of input and label
  xor_generator generator(1000);

  mnist_generator mnist("../../data/mnist/train-images-idx3-ubyte", "../../data/mnist/train-labels-idx1-ubyte");

  // auto& data = mnist.train_inputs();
  /*
   for (auto e : data) {
     std::cout << e << std::endl;
   }
   */
  // run train
  //  net.train<loss_t::mse>(generator.train_inputs(), generator.train_labels(), n_epochs, n_batchsize, adam);
  net.train<loss_t::cross_entropy>(mnist.train_inputs(), mnist.train_labels(), n_epochs, n_batchsize, adam);
}