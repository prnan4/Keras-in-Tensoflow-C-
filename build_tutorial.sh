g++ -c -pipe -g -std=gnu++11 -Wall -W -fPIC -I. -I./ -I./bazel-tensorflow/external/eigen_archive -I./bazel-tensorflow/external/protobuf/src -I./bazel-genfiles -o main.o ./main.cpp
g++  -o Tutorial main.o   -L./bazel-bin/tensorflow -ltensorflow_cc
cp ./bazel-bin/tensorflow/libtensorflow* .