python obfuscation.py --model_name=squeezenet --extra_layer=30 --shortcut=30
pip uninstall -y tensorflow
pip install /tmp/tensorflow_pkg/tensorflow-2.9.1-cp38-cp38-linux_x86_64.whl
python test_obf.py --model_name=squeezenet
