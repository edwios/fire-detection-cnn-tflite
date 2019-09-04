import sys
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

MODEL_NAME = 'firenet'
input_node_names = "InputData/X"
output_node_names = "FullyConnected_2/Softmax"
output_frozen_graph_name = MODEL_NAME+'.pb'
output_optimized_graph_name = 'optimized_'+MODEL_NAME+'.pb'

# Optimize for inference
input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)
    for n in input_graph_def.node:
      print(n.name)
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
                input_graph_def,
                input_node_names.split(","),  # an array of the input node(s)
                output_node_names.split(","), # an array of the output nodes
                tf.float32.as_datatype_enum)

# Save the optimized graph
f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
f.write(output_graph_def.SerializeToString())
