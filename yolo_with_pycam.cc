/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <thread>
#include <memory>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "opencv2/opencv.hpp"
#include "../headers/edgetpu_c.h"

#include "shared_state.h"
#include "camera_thread.h"
#include "inference_thread.h"
#include "polling_thread.h"

#define TFLITE_MINIMAL_CHECK(x)                               \
  if (!(x)) {                                                 \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                  \
  }

// ── Edge TPU 초기화 ───────────────────────────────────────────────────────────
bool setupEdgeTpu(tflite::Interpreter* interpreter) {
    size_t num_devices;

    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

    if (num_devices == 0) {
        std::cerr << "Error: No Edge TPU devices found! Check your connection.\n";
        return false;
    }
    std::cout << "TPU MODE: Found " << num_devices << " device(s).\n";

    const auto& device = devices.get()[0];
    auto* delegate = edgetpu_create_delegate(device.type, device.path,
                                              nullptr, 0);
    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
        std::cerr << "Error: Failed to modify graph with Edge TPU delegate!\n";
        return false;
    }
    std::cout << "Successfully applied Edge TPU delegate to the interpreter.\n";
    return true;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <tflite model> <use_tpu 0|1>\n", argv[0]);
        return 1;
    }
    const char* filename = argv[1];
    const bool  use_tpu  = std::stoi(argv[2]);

    // Force DISPLAY to localhost:10.0 for X11 forwarding convenience
    setenv("DISPLAY", "localhost:10.0", 1);

    // ── (1) 런타임 카메라 자동 감지 ──────────────────────────────────────────
    static constexpr int MAX_PROBE_INDEX = 8;
    std::vector<int>              cam_indices;
    std::vector<cv::VideoCapture> caps;

    printf("[init] Probing cameras (index 0 ~ %d)...\n", MAX_PROBE_INDEX - 1);
    for (int idx = 0; idx < MAX_PROBE_INDEX; ++idx) {
        cv::VideoCapture probe(idx);
        if (!probe.isOpened()) continue;

        cv::Mat test_frame;
        probe >> test_frame;
        probe.release();
        if (test_frame.empty()) continue;   // 메타 장치 등 필터링

        printf("[init] Camera %d detected at /dev/video%d\n",
               (int)cam_indices.size(), idx);
        cam_indices.push_back(idx);
        caps.emplace_back(idx);
    }

    const int num_cameras = (int)cam_indices.size();
    if (num_cameras == 0) {
        fprintf(stderr, "Error: No usable camera found.\n");
        return 1;
    }
    printf("[init] Total %d camera(s) available.\n", num_cameras);

    // ── (2) 카메라 활성화 플래그 및 입력 슬롯 초기화 ──────────────────────────
    for (int i = 0; i < num_cameras; ++i) {
        camera_active_flags.emplace_back(new std::atomic<bool>(true));
        input_slots.emplace_back(new CameraSlot());
    }

    // ── (3) TFLite 모델 로드 & 인터프리터 빌드 ───────────────────────────────
    auto model = tflite::FlatBufferModel::BuildFromFile(filename);
    TFLITE_MINIMAL_CHECK(model != nullptr);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    // ── (4) Edge TPU 설정 ─────────────────────────────────────────────────────
    if (use_tpu) {
        if (!(tpu_mode = setupEdgeTpu(interpreter.get()))) {
            std::cerr << "Falling back to CPU mode...\n";
        }
    }

    // ── (5) 텐서 버퍼 할당 & 모델 정보 출력 ─────────────────────────────────
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    printf("=== Pre-invoke Interpreter State ===\n");
    for (int i = 0; i < (int)interpreter->inputs().size(); ++i) {
        TfLiteTensor* t = interpreter->tensor(interpreter->inputs()[i]);
        printf("Input [%d]: %s  Type: %d  Dims:", i, t->name, t->type);
        for (int d = 0; d < t->dims->size; ++d) printf(" %d", t->dims->data[d]);
        printf("\n");
    }
    for (int i = 0; i < (int)interpreter->outputs().size(); ++i) {
        TfLiteTensor* t = interpreter->tensor(interpreter->outputs()[i]);
        printf("Output[%d]: %s  Type: %d  Dims:", i, t->name, t->type);
        for (int d = 0; d < t->dims->size; ++d) printf(" %d", t->dims->data[d]);
        printf("\n");
    }

    TfLiteTensor* in = interpreter->tensor(interpreter->inputs()[0]);
    const int input_height = in->dims->data[1];
    const int input_width  = in->dims->data[2];
    const int channels     = in->dims->data[3];
    const int total_pixels = input_height * input_width * channels;

    // ── (6) 스레드 시작 ───────────────────────────────────────────────────────
    std::thread polling_thread(camera_status_polling_func);

    std::vector<std::thread> camera_threads;
    for (int i = 0; i < num_cameras; ++i) {
        camera_threads.emplace_back(camera_thread_func,
                                    &caps[i], i,
                                    input_width, input_height);
    }

    std::thread inference_thread(inference_thread_func,
                                  interpreter.get(), total_pixels);

    // ── (7) 메인 스레드: 시각화 루프 ─────────────────────────────────────────
    while (running) {
        DetectionResult result;
        {
            std::unique_lock<std::mutex> lock(output_mutex);
            // 결과가 올 때까지 최대 16ms(≈60fps) 대기
            output_cv.wait_for(lock, std::chrono::milliseconds(16),
                               [] { return !output_queue.empty() || !running; });

            if (output_queue.empty()) {
                if (getenv("DISPLAY")) {
                    if (cv::waitKey(1) == 'q') { running = false; break; }
                }
                continue;
            }
            result = output_queue.front();
            output_queue.pop();
        }

        yolo_output_visualize(result.display_image, result.detections);

        cv::Mat show;
        cv::resize(result.display_image, show, cv::Size(300, 300));
        const std::string win = "Yolo cam " + std::to_string(result.camera_id);

        if (getenv("DISPLAY")) {
            try { cv::imshow(win, show); } catch (...) {}
            if (cv::waitKey(1) == 'q') { running = false; break; }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // ── (8) 정리 ──────────────────────────────────────────────────────────────
    running = false;
    input_cv.notify_all();
    output_cv.notify_all();

    if (polling_thread.joinable())   polling_thread.join();
    for (auto& t : camera_threads)  if (t.joinable()) t.join();
    if (inference_thread.joinable()) inference_thread.join();

    for (auto& cap : caps) cap.release();
    cv::destroyAllWindows();
    return 0;
}
