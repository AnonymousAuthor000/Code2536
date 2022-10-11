# Code2536

Code for the submission Code2536

1. Preparation

(1) The dependency can be found in `environment.yml`. To create the conda environment:

`conda env create -f environment.yml`

`conda activate code2536`

Install the Flatbuffer:

`conda install -c conda-forge flatbuffers`

(2) Download the source code of the TensorFlow. Here we test our tool on v2.9.1.

`wget https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.9.1.zip`

Unzip the file:

`unzip v2.9.1`

(3) Download the Bazel:

`wget https://github.com/bazelbuild/bazelisk/releases/download/v1.14.0/bazelisk-linux-amd64`

`chmod +x bazelisk-linux-amd64`

`sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel`

You can test the Bazel:

`which bazel`

It should return:

`bazel is /usr/local/bin/bazel`

(4) Configure the build:

`cd tensorflow-2.9.1/`

`./configure`

`cd ..`

You can use the default setting in all options.

(5) Copy the 'BUILD' and 'register.cc' to the source code:  

`cp ./files/* ./tensorflow-2.9.1/tensorflow/lite/kernels/`

2. Test

`bash ./build_tfl.sh`
