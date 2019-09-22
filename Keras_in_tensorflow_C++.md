# Keras in tensorflow C++

## Testing the model

Download Model data_model.h5 and use its path. Also use the path of the test image.
Save test.py and run the program as follows. 28 here corresponds to the size of the test image.

```sh
python test.py -m '/Users/nandhinee_pr/admatic/Cnn Models/data_model.h5' -i '/Users/nandhinee_pr/Desktop/ecaimg3.jpg' -s 28
```

## To convert H5 file to .pb file

Save the code convert.py. This code converts the H5 model file to .pb file. To run the code do:

```sh
python convert.py '/Users/nandhinee_pr/admatic/Cnn Models/data_model.h5' -n 1 -o '/Users/nandhinee_pr/Desktop' -p "out"
```
This code creates output_graph.pb file and Saves the graph in ASCII format. Input and output node name are retrieved from the ASCII file.

## Testing the model with tensorflow 

Save the code label_image.py. --labels uses location of the labels.txt file.  Width and height of the test image used is 28. Location of the test image has to be specified. Name of the input layer and output layer is taken from the ASCII file.

```sh
python label_image.py --graph='/Users/nandhinee_pr/Desktop/output_graph.pb' --labels=/Users/nandhinee_pr/Desktop/labels.txt --input_width=28 --input_height=28 --input_layer=conv2d_3_input  --output_layer=out_0 --image=/Users/nandhinee_pr/Desktop/ecaimg3.jpg
```

## Bazel Installation

Tensorflow version used = 1.4.0
Bazel version used = 0.5.4

After installing the specified bazel version, to configure:

```sh
sudo ./configure
```

To build:

```sh
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 //tensorflow:libtensorflow_cc.so
```

Running this script creates bazel-bin, bazel_tensorflow, bazel_out and bazel_gen files.

## To run C++ code

Save main.cpp and build_tutorial.sh in the same directory.
Running the shell script creates output main.o.

```sh
 sudo ./build_tutorial.sh 
 ```

Running this script  invokes the cpp file and shows the predictions.
```sh
 ./Tutorial --graph='/Users/nandhinee_pr/Desktop/output_graph.pb' --labels=/Users/nandhinee_pr/Desktop/labels.txt --input_width=28 --input_height=28 --input_layer=conv2d_3_input  --output_layer=out_0 --image=/Users/nandhinee_pr/Desktop/ecaimg3.jpg
 ```
