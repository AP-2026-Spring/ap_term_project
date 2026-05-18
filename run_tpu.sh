#!/bin/bash

# .env 파일이 존재하면 읽어서 환경 변수로 적용
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

./yolo_with_pycam best_full_integer_quant_edgetpu.tflite 1
