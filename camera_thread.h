#pragma once
#include "opencv2/opencv.hpp"

// ── Thread A (Producer): 카메라 캡처 & 전처리 & 하드웨어 On/Off 제어 ──────────
// cap      : 이 스레드가 담당하는 VideoCapture 객체
// camera_id: 카메라 식별자 (camera_active_flags 인덱스와 동일)
// input_width, input_height: TFLite 모델 입력 해상도
void camera_thread_func(cv::VideoCapture* cap, int camera_id,
                        int input_width, int input_height);
