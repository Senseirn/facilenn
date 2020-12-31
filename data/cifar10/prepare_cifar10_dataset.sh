#/bin/sh

cd `dirname $0`

echo "preparing cifar10 dataset.."

#1. download mnist dataset
echo "downloading datasets.."
curl -OLs https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz > /dev/null
echo "download finished.."

#2. extract them
echo "extracting datasets.."
tar xvf *.gz
mv cifar-10-batches-bin/* .
echo "extraction finished.."
echo "complete!"