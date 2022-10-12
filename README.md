# Code2536

Code for the submission Code2536

## Preparation

0. Download the code:

`git clone https://github.com/AnonymousAuthor000/Code2536.git`

`cd Code2536`

1. The dependency can be found in `environment.yml`. To create the conda environment:

`conda env create -f environment.yml`

`conda activate code2536`

Install the Flatbuffer:

`conda install -c conda-forge flatbuffers`

(if no npm) install the npm:

`sudo apt-get install npm`

Install the jsonrepair:

`npm install -g jsonrepair`


2. Download the source code of the TensorFlow. Here we test our tool on v2.9.1.

`wget https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.9.1.zip`

Unzip the file:

`unzip v2.9.1`

3. Download the Bazel:

`wget https://github.com/bazelbuild/bazelisk/releases/download/v1.14.0/bazelisk-linux-amd64`

`chmod +x bazelisk-linux-amd64`

`sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel`

You can test the Bazel:

`which bazel`

It should return:
`/usr/local/bin/bazel`

4. Configure the build:

`cd tensorflow-2.9.1/`

`./configure`

`cd ..`

You can use the default setting in all options.

5. Copy the congigurations and script to the source code:  

`cp ./files/kernel_files/* ./tensorflow-2.9.1/tensorflow/lite/kernels/`

`cp ./files/build_files/build.sh ./tensorflow-2.9.1/`

Note that you can mofify the maximal number of jobs in the 'build.sh' script. Here I set it as `--jobs=14`. 

## Test

1. Build the obfuscation library:

`python obfuscation.py --extra_layer=30 --shortcut=30`

2. Reinstall the TensorFlow:

`pip uninstall -y tensorflow`

`pip install /tmp/tensorflow_pkg/tensorflow-2.9.1-cp38-cp38-linux_x86_64.whl`

3. Generate the obfuscated model:

`python generate_obf.py`

Then, the obfuscated model will be generated as 'obf_model.tflite' file. You can visualize the difference between the original model and obfuscated model by Netron (https://netron.app/).

4. test the obfuscated model:

`python test_model.py`
