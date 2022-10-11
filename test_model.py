import os
import sys
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from model_assembler import *

from utils.utils import *
import time

def model_inference(interpreter, inputs):
    for i in range(len(inputs)):
        interpreter.set_tensor(input_details[0]["index"], np.expand_dims(inputs[i], 0))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
    return output

model_path = './tflite_model/fruit.tflite'

# --------------------------------------------------
# generate random data
# --------------------------------------------------
inputs = generate_random_data(model_path, batch_size=1000)[0]
# print(inputs[0].shape)
x = tf.constant(np.expand_dims(inputs[0], 0), dtype=tf.float32)

# --------------------------------------------------
# get the output of the obfuscated model
# --------------------------------------------------
interpreter = tf.lite.Interpreter(
 'obf_model.tflite', experimental_preserve_all_tensors=True
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

time_start=time.time()
output_obf = model_inference(interpreter, inputs)
time_end=time.time()
print('obf time cost: ',time_end-time_start,'s')
gc.collect()

# --------------------------------------------------
# get the output of the original model
# --------------------------------------------------
interpreter = tf.lite.Interpreter(
 model_path, experimental_preserve_all_tensors=True
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

time_start=time.time()
output_ori = model_inference(interpreter, inputs)
time_end=time.time()
print('ori time cost: ',time_end-time_start,'s')
gc.collect()

print('obfuscation error:', (output_obf.squeeze()-output_ori.squeeze()).sum())
