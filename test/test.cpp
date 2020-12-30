#include "../tino/tino.h"

int main() {
  using namespace tino;
  using namespace tino::layers;
  using namespace tino::core;
  using namespace tino::utils;

  // define network
  tino::network net;
  net.add(new fully_connected_layer(2, 8));
  net.add(new relu_layer());
  net.add(new fully_connected_layer(8, 8));
  net.add(new relu_layer());
  net.add(new fully_connected_layer(8, 1));
  net.add(new relu_layer());

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
  int n_epochs = 1;
  int n_batchsize = 100;

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
  net.train(generator.train_inputs(), generator.train_labels(), n_epochs, n_batchsize, adam);
  //  net.train(mnist.train_inputs(), mnist.train_labels(), n_epochs, n_batchsize, adam);
}