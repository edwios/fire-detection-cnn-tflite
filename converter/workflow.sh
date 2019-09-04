#!/bin/bash

# Workflow

# Prepare model protobuf
python3 firenet-to-protobuf.py
# Output: firenet.pb

# Convert protobuf to tflite (transition)
# Output: firenet.tflite
python3 protobuf-to-tflite.py

# Clean up tflite for quantisation
# Output: optimized_firenet.tflite
python3 optimize_protobuf.py

# Perform Quantisation to Edge TPU
# Output: firenet.tflite
bash < quantize.sh


