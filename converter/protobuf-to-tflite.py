import tensorflow as tf

graph_def_file = "optimized_firenet.pb"
input_arrays = ["InputData/X"]
output_arrays = ["FullyConnected_2/Softmax"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file, input_arrays, output_arrays, {"InputData/X":[None, 224, 224, 3]})
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
open("firenet.tflite", "wb").write(tflite_model)

