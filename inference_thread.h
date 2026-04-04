#pragma once
#include "tensorflow/lite/interpreter.h"

// ── Thread B (Consumer): 추론 및 파싱 ────────────────────────────────────────
// interpreter : TFLite Interpreter (main에서 생성, 이 스레드 전용 사용)
// total_pixels: input_height × input_width × channels
void inference_thread_func(tflite::Interpreter* interpreter, int total_pixels);
