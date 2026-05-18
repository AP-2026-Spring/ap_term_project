#!/bin/bash

# .env ?뚯씪??議댁옱?섎㈃ ?쎌뼱???섍꼍 蹂?섎줈 ?곸슜
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

./yolo_with_pycam best_full_integer_quant_edgetpu.tflite 1
