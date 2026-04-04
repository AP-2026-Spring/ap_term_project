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
    while (running) {
        // ── input_queue 에서 FrameData 꺼내기 ──────────────────────────────
        FrameData frame;
        {
            std::unique_lock<std::mutex> lock(input_mutex);
            // 큐가 빌 때 최대 100ms 블록 대기 (Spinlock 방지)
            input_cv.wait_for(lock, std::chrono::milliseconds(100),
                              [] { return !input_queue.empty() || !running; });

            if (input_queue.empty()) continue;  // 타임아웃 또는 종료 신호

            frame = input_queue.front();
            input_queue.pop();
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
            output_queue.push({frame.camera_id, display.clone(), detections});
        }
        output_cv.notify_one();
    }
}
