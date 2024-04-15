#!/bin/sh

cd $(dirname $0)

echo "preparing mnist dataset.."

#1. download mnist dataset
echo "downloading datasets.."
curl -OLs http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz >/dev/null
curl -OLs http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz >/dev/null
curl -OLs http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz >/dev/null
curl -OLs http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz >/dev/null
echo "download finished.."

#2. extract them
echo "extracting datasets.."
gzip -f -d *.gz
echo "extraction finished.."
echo "complete!"
