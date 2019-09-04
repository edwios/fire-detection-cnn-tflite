#!/bin/bash

tflite_convert --output_file=firenet.tflite --graph_def_file=optimized_firenet.pb --inference_type=QUANTIZED_UINT8 --input_arrays='InputData/X' --output_arrays='FullyConnected_2/Softmax' --inference_type=QUANTIZED_UINT8 --post_training_quantize --input_shapes=1,224,224,3 --mean_values=128 --std_dev_values=127

