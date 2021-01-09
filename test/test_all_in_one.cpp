#include "../tino/tino.h"

#include <chrono>

int main() {
  using namespace tino;
  using namespace tino::layers;
  using namespace tino::core;
  using namespace tino::backends;
  using namespace tino::utils;
  using namespace tino::loss;

  {
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

    // run 10 epochs with batch_size 200
    int n_epochs    = 2;
    int n_batchsize = 200;

    mnist_loader mnist("../../data/mnist/train-images-idx3-ubyte", "../../data/mnist/train-labels-idx1-ubyte");

    std::chrono::system_clock::time_point start, end;
    context ctx(backend_t::naive, parallelize_t::none);
    {
      // run train with backend=naive, parallelize=none
      net.train<loss_t::cross_entropy>(mnist.train_inputs(), mnist.train_labels(), n_epochs, n_batchsize, adam, ctx);
    }

    { // run train with backend=naive, parallelize=openmp
      ctx.parallelize(parallelize_t::openmp);
      net.train<loss_t::cross_entropy>(mnist.train_inputs(), mnist.train_labels(), n_epochs, n_batchsize, adam, ctx);
    }

    {
      // run train with backend=naive, parallelize=intel_tbb
      ctx.parallelize(parallelize_t::intel_tbb);
      net.train<loss_t::cross_entropy>(mnist.train_inputs(), mnist.train_labels(), n_epochs, n_batchsize, adam, ctx);
    }

    ctx.backend(backend_t::openblas);
    ctx.parallelize(parallelize_t::none);
    { // run train with backend=openblas, parallelize=none
      ctx.parallelize(parallelize_t::none);
      net.train<loss_t::cross_entropy>(mnist.train_inputs(), mnist.train_labels(), n_epochs, n_batchsize, adam, ctx);
    }

    {
      // run train with backend=openblas, parallelize=openmp
      ctx.parallelize(parallelize_t::openmp);
      net.train<loss_t::cross_entropy>(mnist.train_inputs(), mnist.train_labels(), n_epochs, n_batchsize, adam, ctx);
    }

    { // run train with backend=openblas, parallelize=intel_tbb
      ctx.parallelize(parallelize_t::intel_tbb);
      net.train<loss_t::cross_entropy>(mnist.train_inputs(), mnist.train_labels(), n_epochs, n_batchsize, adam, ctx);
    }
  }
}