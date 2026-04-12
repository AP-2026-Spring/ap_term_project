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
        FrameData frame_data; // 로직 유지를 위해 내부적으로 잠시 사용
        bool found = false;

        {
            std::unique_lock<std::mutex> lock(input_cv_mutex);
            // 모든 슬롯을 확인해서 처리 대기 중인 프레임이 있는지 체크
            auto check_ready = [&]() {
                for (int i = 0; i < num_cameras; ++i) {
                    int idx = (last_cam_id + 1 + i) % num_cameras;
                    if (input_slots[idx]->ready) {
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

            if (input_slots[last_cam_id]->ready) {
                std::lock_guard<std::mutex> slot_lock(input_slots[last_cam_id]->mtx);
                frame_data.camera_id = last_cam_id;
                frame_data.image     = input_slots[last_cam_id]->image.clone();
                input_slots[last_cam_id]->ready = false;
                found = true;
            }
        }

        if (!found) continue;

        // 원활한 컴파일을 위해 내부 변수명을 frame_data -> frame 으로 매칭
        auto& frame = frame_data;

        // ── TFLite 입력 텐서에 데이터 복사 ──────────────────────────────────
        TfLiteTensor* input_tensor_ptr =
            interpreter->tensor(interpreter->inputs()[0]);

        if (tpu_mode) {
            if (input_tensor_ptr->type == kTfLiteUInt8) {
                uint8_t* dst = interpreter->typed_input_tensor<uint8_t>(0);
                std::memcpy(dst, frame.image.data, total_pixels);
            } else if (input_tensor_ptr->type == kTfLiteInt8) {
                int8_t* dst = interpreter->typed_input_tensor<int8_t>(0);
                for (int i = 0; i < total_pixels; ++i)
                    dst[i] = (int8_t)((int)frame.image.data[i] - 128);
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
        printf("\r[cam %d] Inference Latency: %.2f ms   ",
               frame.camera_id, latency.count());
        fflush(stdout);

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
