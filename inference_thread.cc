#include "inference_thread.h"
#include "shared_state.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>
#include <chrono>
#include "tensorflow/lite/c/common.h"

// ── Thread B (Consumer): 추론 및 파싱 ────────────────────────────────────────
//  condition_variable 로 input_queue 에 데이터가 생길 때까지 블록 (Spinlock 방지)
//  → TFLite Invoke() → 박스 파싱 → output_queue.push → output_cv.notify
void inference_thread_func(tflite::Interpreter* interpreter, int total_pixels) {
    int last_cam_id = -1;
    const int num_cameras = (int)input_slots.size();

    while (running) {
        // ── input_slots 에서 최신 프레임 공정하게(Round-robin) 꺼내기 ──────────
        FrameData frame; // 로직 유지를 위해 내부적으로 잠시 사용
        bool found = false;

        {
            std::unique_lock<std::mutex> lock(input_cv_mutex);
            // 모든 슬롯을 확인해서 처리 대기 중인 프레임이 있는지 체크
            auto check_ready = [&]() {
                for (int i = 0; i < num_cameras; ++i) {
                    int idx = (last_cam_id + 1 + i) % num_cameras;
                    
                    bool is_ready = false;
                    {
                        // ready 상태를 읽을 때도 반드시 해당 슬롯의 mutex를 잠가야 최신 값을 보장받음
                        std::lock_guard<std::mutex> slot_lock(input_slots[idx]->mtx);
                        is_ready = input_slots[idx]->ready;
                    }

                    if (is_ready) {
                        last_cam_id = idx;
                        return true;
                    }
                }
                return false;
            };

            if (!check_ready()) {
                input_cv.wait_for(lock, std::chrono::milliseconds(100),
                                  [&] { return check_ready() || !running; });
            }

            if (!running) break;

            // 데이터를 가져올 때도 lock을 먼저 걸고 ready 상태를 재확인
            // (초기 last_cam_id가 -1일 때의 Out-of-bounds 방지 포함)
            if (last_cam_id >= 0) {
                std::lock_guard<std::mutex> slot_lock(input_slots[last_cam_id]->mtx);
                if (input_slots[last_cam_id]->ready) {
                    frame.camera_id = last_cam_id;
                    frame.image     = input_slots[last_cam_id]->image.clone();
                    input_slots[last_cam_id]->ready = false;
                    found = true;
                }
            }
        }

        if (!found) continue;

        // 100% 확실히 꺼진 카메라에 대해서는 추론을 즉시 스킵하고 프레임 폐기
        if (frame.camera_id >= 0 && frame.camera_id < (int)camera_active_flags.size()) {
            if (!camera_active_flags[frame.camera_id]->load()) {
                continue; // 추론 스킵
            }
        }

        // ── TFLite 입력 텐서에 데이터 복사 ──────────────────────────────────
        TfLiteTensor* input_tensor_ptr =
            interpreter->tensor(interpreter->inputs()[0]);

        if (tpu_mode) {
            if (input_tensor_ptr->type == kTfLiteUInt8) {
                uint8_t* dst = interpreter->typed_input_tensor<uint8_t>(0);
                std::memcpy(dst, frame.image.data, total_pixels);
            } else if (input_tensor_ptr->type == kTfLiteInt8) {
                int8_t* dst = interpreter->typed_input_tensor<int8_t>(0);
                float in_scale = input_tensor_ptr->params.scale;
                int in_zp = input_tensor_ptr->params.zero_point;
                
                for (int i = 0; i < total_pixels; ++i) {
                    // [0~255] -> [0~1] -> Quantized
                    float normalized = (float)frame.image.data[i] / 255.0f;
                    dst[i] = (int8_t)(normalized / in_scale + in_zp);
                }
            }
        } else {
            float* dst = interpreter->typed_input_tensor<float>(0);
            for (int i = 0; i < total_pixels; ++i)
                dst[i] = (float)frame.image.data[i] / 255.0f;
        }

        // ── TFLite Invoke() ───────────────────────────────────────────────
        auto t0 = std::chrono::high_resolution_clock::now();
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "[cam " << frame.camera_id << "] Inference failed!\n";
            continue;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> latency = t1 - t0;

        // ── 박스 파싱 ──────────────────────────────────────────────────────
        auto detections = parse_detections_thread_safe(
            interpreter, frame.image.cols, frame.image.rows);

        // 시각화용 BGR 이미지 생성 (RGB → BGR)
        cv::Mat display;
        cv::cvtColor(frame.image, display, cv::COLOR_RGB2BGR);

        // ── output_queue 적재 & 메인 스레드 알림 ─────────────────────────
        {
            std::lock_guard<std::mutex> lock(output_mutex);
            if (output_queue.size() >= OUTPUT_QUEUE_MAX) {
                output_queue.pop();
            }
            // aggregate initialization 대신 명시적 생성자 호출 (컴파일러 호환성)
            DetectionResult res;
            res.camera_id = frame.camera_id;
            res.display_image = display.clone();
            res.detections = detections;
            output_queue.push(res);
        }
        output_cv.notify_one();
    }
}
