#include "../tino/tino.h"

int main() {
  using namespace tino;
  using namespace tino::layers;
  using namespace tino::core;
  using namespace tino::backends;
  using namespace tino::utils;
  using namespace tino::loss;

  // define network
  tino::network net;
  net.add(new fully_connected_layer(1 * 28 * 28, 16));
  net.add(new relu_layer());
  net.add(new fully_connected_layer(16, 16));
  net.add(new relu_layer());
  net.add(new fully_connected_layer(16, 10));
  net.add(new softmax_layer());

  // define how to initialize weights
  // here we use He initializer
  net.weight_initializer(initializers::He<TINO_FLOAT_TYPE>);

  // declare optimizer
  // here we use Adam
  optimizers::Adam adam;
  adam.alpha(0.001f);

  // run 10 epochs with batch_size=10
  int n_epochs = 1;
  int n_batchsize = 200;

  // generate xor dataset which contains 1,000 pairs of input and label
  // xor_loader generator(1000);

  mnist_loader mnist("../../data/mnist/train-images-idx3-ubyte", "../../data/mnist/train-labels-idx1-ubyte");

  //cifar10_loader cifar10("../../data/cifar10");

  context ctx(backend_t::openblas, parallelize_t::intel_tbb);

  // run train
  //  net.train<loss_t::mse>(generator.train_inputs(), generator.train_labels(), n_epochs, n_batchsize, adam);
  net.train<loss_t::cross_entropy>(mnist.train_inputs(), mnist.train_labels(), n_epochs, n_batchsize, adam, ctx);
  // net.train<loss_t::cross_entropy>(cifar10.train_inputs(), cifar10.train_labels(), n_epochs, n_batchsize, adam);
}