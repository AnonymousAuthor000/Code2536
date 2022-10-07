import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import json
import random
import argparse
# import orjson
# from tensorflow import keras
from model_parser import *

from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--extra_layer', type=int, default=0, help='number extra layer')
parser.add_argument('--shortcut', type=int, default=0, help='number shortcut')
opt = parser.parse_args()

def reduce_size_json(json_file):
    with fileinput.input(files=json_file, inplace=True) as f:
        keep_sign = True
        for line in f:
            if 'buffers:' in line:
                keep_sign = False
                print('}', end="")
            if keep_sign:
                print(line, end="")


def random_shortcut(model_json):
    num_op = len(model_json['subgraphs'][0]["operators"])
    for i in range(opt.shortcut):
        rand_shortcut_pair = random.sample(range(0, num_op), 2)
        rand_shortcut_pair.sort()
        op_i = model_json['subgraphs'][0]["operators"][rand_shortcut_pair[1]]
        if 'builtin_options_type' in op_i.keys():
            if op_i['builtin_options_type'] != 'ConcatenationOptions' and op_i['builtin_options_type'] != 'AddOptions':
        # if op_i['builtin_options_type'] != 'ConcatenationOptions':
                op_i["inputs"].append(model_json['subgraphs'][0]["operators"][rand_shortcut_pair[0]]["outputs"][0])


def random_extra(model_json, out_start_point):
    num_op = len(model_json['subgraphs'][0]["operators"])
    for i in range(opt.extra_layer):
        rand_extra_pair = random.sample(range(0, num_op), 2)
        rand_extra_pair.sort()
        op_i = model_json['subgraphs'][0]["operators"][rand_extra_pair[1]]
        if 'builtin_options_type' in op_i.keys():
            if op_i['builtin_options_type'] != 'ConcatenationOptions' and op_i['builtin_options_type'] != 'AddOptions':
                op_i["inputs"].append(out_start_point+i)
                model_json['subgraphs'][0]["operators"].append({'builtin_options_type': 'ObfOptions', "inputs": [model_json['subgraphs'][0]["operators"][rand_extra_pair[0]]["outputs"][0]], "outputs": [out_start_point+i]})


model_path = './tflite_model/'
model_name = 'squeezenet.tflite'
interpreter = tf.lite.Interpreter(
 os.path.join(model_path, model_name)
)
interpreter.allocate_tensors()
# --------------------------------------------------
# parse the TFLite model and generate obfuscated op
# --------------------------------------------------
os.system('flatc -t schema.fbs -- %s' % os.path.join(model_path, model_name))
reduce_size_json(os.path.splitext(model_name)[0] + '.json')
os.system('jsonrepair %s.json --overwrite' % os.path.splitext(model_name)[0])
# for op in interpreter._get_ops_details():
#     print(op)

with open('%s.json' % os.path.splitext(model_name)[0],'r') as f:
    model_json_f = f.read()
model_json = json.loads(model_json_f)

# op_details = interpreter._get_ops_details()
# print(op_details)

random_shortcut(model_json)
tensor_list = []
for input in interpreter.get_input_details():
    tensor_list.append(input['index'])
for tensor_details in interpreter.get_tensor_details():
    tensor_list.append(tensor_details["index"])
tensor_list.sort()
random_extra(model_json, tensor_list[-1]+10)


inout_list = []
for i in range(len(model_json['subgraphs'][0]["operators"])):
    # print(model_json['subgraphs'][0]["operators"][i])
    for j in range(len(model_json['subgraphs'][0]["operators"][i]['outputs'])):
        inout_list.append(model_json['subgraphs'][0]["operators"][i]['outputs'][j])

for input in interpreter.get_input_details():
    inout_list.append(input['index'])

# for output in interpreter.get_output_details():
#     inout_list.append(output['index'])

jsontext = lib_generator(model_json, interpreter, inout_list)

# --------------------------------------------------
# build the custom library
# --------------------------------------------------
currentPath = os.getcwd().replace('\\','/')
os.chdir('./tensorflow-2.9.1/')
os.system("bash build.sh")
os.chdir(currentPath)
# print(inout_list)
for op in jsontext['oplist']:
    del_list = []
    # print("input:", op['input'])
    for i in range(len(op['input'])):
        if not (op['input'][i] in inout_list):
            # print("not in inout_list:", op['input'][i])
            del_list.append(op['input'][i])
    # print("del:", del_list)
    for j in range(len(del_list)):
        op['input'].remove(del_list[j])

jsondata = json.dumps(jsontext,indent=4,separators=(',', ': '))
file = open('./ObfusedModel.json', 'w')
file.write(jsondata)
file.close()

