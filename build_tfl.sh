python obfuscation.py --extra_layer=30 --shortcut=30
# sleep 3
pip uninstall -y tensorflow
pip install /tmp/tensorflow_pkg/tensorflow-2.9.1-cp38-cp38-linux_x86_64.whl
# sleep 3
python -m memory_profiler test_obf.py
# python test_obf.py