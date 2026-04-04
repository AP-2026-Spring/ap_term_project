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
#include <cstring>
#include <vector>
#include <queue>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "opencv2/opencv.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "yolo_with_pycam.h"
#include "../headers/edgetpu_c.h"
//#include <raspicam/raspicam_cv.h>

#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

using namespace std;

// ── Static member definitions for yolo::YOLO_Parser ──────────────────────────
std::vector<std::vector<float>> yolo::YOLO_Parser::real_bbox_cls_vector;
std::vector<int>                yolo::YOLO_Parser::real_bbox_cls_index_vector;
std::vector<std::vector<int>>   yolo::YOLO_Parser::real_bbox_loc_vector;
std::vector<yolo::YOLO_Parser::BoundingBox> yolo::YOLO_Parser::result_boxes;

// ── [1] Producer → Consumer 전달 구조체 & 큐 ──────────────────────────────────
// FrameData: 어느 카메라(camera_id)에서 온 전처리 완료 프레임인지를 함께 보관
struct FrameData {
    int     camera_id;
    cv::Mat image;      // 전처리(resize + cvtColor) 완료된 RGB 이미지
};

// Producer(Thread A) → Consumer(Thread B) 큐
std::queue<FrameData>  input_queue;
std::mutex             input_mutex;         // input_queue 보호
std::condition_variable input_cv;           // 추론 스레드 대기 알림용

// ── [2] Consumer → Main Thread 전달 구조체 & 큐 ──────────────────────────────
// DetectionResult: 추론/파싱 결과 + 카메라 ID + 시각화용 이미지
struct DetectionResult {
    int                    camera_id;
    cv::Mat                display_image;   // BGR 변환된 시각화용 이미지
    std::vector<Detection> detections;
};

// Consumer(Thread B) → Main Thread 큐
std::queue<DetectionResult> output_queue;
std::mutex                  output_mutex;   // output_queue 보호
std::condition_variable     output_cv;      // 메인 스레드 대기 알림용

// ── [3] 카메라 활성화 플래그 ────────────────────────────────────────────────
// camera_active_flags[i]: i번 카메라의 On(true) / Off(false) 상태
// std::vector<std::atomic<bool>>은 복사 불가이므로 unique_ptr 배열로 관리
static constexpr int NUM_CAMERAS = 2;
std::unique_ptr<std::atomic<bool>> camera_active_flags[NUM_CAMERAS];

std::atomic<bool> running{true};

// 큐 최대 깊이 - 메모리 보호
static constexpr size_t INPUT_QUEUE_MAX  = 4;
static constexpr size_t OUTPUT_QUEUE_MAX = 4;

// 상태 파일 경로 (RAM 디스크 - SD 카드 수명 보호)
static constexpr const char* CAM_CMD_PATH = "/dev/shm/camera_cmd.txt";

#define TFLITE_MINIMAL_CHECK(x)                               \
  if (!(x)) {                                                 \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                  \
  }

bool tpu_mode = false;

// ── Edge TPU 초기화 (핵심 로직 변경 없음) ────────────────────────────────────
bool setupEdgeTpu(tflite::Interpreter* interpreter) {
    size_t num_devices;

    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

    if (num_devices == 0) {
        std::cerr << "Error: No Edge TPU devices found! Check your connection."
                  << std::endl;
        return false;
    }
    std::cout << "TPU MODE: Found " << num_devices << " device(s)." << std::endl;

    const auto& device = devices.get()[0];
    auto* delegate = edgetpu_create_delegate(device.type, device.path,
                                              nullptr, 0);

    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
        std::cerr << "Error: Failed to modify graph with Edge TPU delegate!"
                  << std::endl;
        return false;
    }
    std::cout << "Successfully applied Edge TPU delegate to the interpreter."
              << std::endl;
    return true;
}

// ── Polling Thread: 공유 파일 폴링으로 카메라 On/Off 제어 ─────────────────────
//  1초마다 /dev/shm/camera_cmd.txt 를 읽어 camera_active_flags 갱신
//  파일 포맷: "0:1\n1:0\n" → 0번 카메라 ON, 1번 카메라 OFF
void camera_status_polling_func() {
    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        try {
            std::ifstream ifs(CAM_CMD_PATH);
            if (!ifs.is_open()) {
                // 파일 없으면 조용히 넘어감
                continue;
            }

            std::string line;
            while (std::getline(ifs, line)) {
                if (line.empty()) continue;

                // 포맷 파싱: "cam_id:state" (예: "0:1", "1:0")
                std::istringstream ss(line);
                std::string token_id, token_state;

                if (!std::getline(ss, token_id,   ':')) continue;
                if (!std::getline(ss, token_state, ':')) continue;

                int  cam_id = -1;
                int  state  = -1;
                try {
                    cam_id = std::stoi(token_id);
                    state  = std::stoi(token_state);
                } catch (...) {
                    // 정수 변환 실패 → 해당 줄 무시
                    continue;
                }

                if (cam_id < 0 || cam_id >= NUM_CAMERAS) continue;
                if (state != 0 && state != 1)             continue;

                bool new_active = (state == 1);
                bool old_active = camera_active_flags[cam_id]->load();

                if (old_active != new_active) {
                    camera_active_flags[cam_id]->store(new_active);
                    printf("[polling] camera %d → %s\n",
                           cam_id, new_active ? "ON" : "OFF");
                    fflush(stdout);
                }
            }
        } catch (const std::exception& e) {
            // ifstream 관련 예외 → 프로그램 중단 없이 다음 주기에 재시도
            fprintf(stderr, "[polling] file I/O exception: %s\n", e.what());
        } catch (...) {
            fprintf(stderr, "[polling] unknown file I/O exception\n");
        }
    }
}

// ── Thread A (Producer): 카메라 캡처 & 전처리 & 하드웨어 On/Off 제어 ──────────
//  · flag=false & cap.isOpened() → cap.release()
//  · flag=true  & !cap.isOpened() → cap.open(camera_id) (실패 시 재시도)
//  · flag=false 동안은 100ms Idle (CPU 보호)
//  · 캡처 성공 시 resize → cvtColor → input_queue.push
void camera_thread_func(cv::VideoCapture* cap, int camera_id,
                        int input_width, int input_height) {
    while (running) {
        bool is_active = camera_active_flags[camera_id]->load();

        // ── 하드웨어 On/Off 전환 처리 ──────────────────────────────────────
        if (!is_active) {
            // OFF 명령: 열려 있으면 해제
            if (cap->isOpened()) {
                cap->release();
                printf("[cam %d] released (OFF)\n", camera_id);
                fflush(stdout);
            }
            // Idle 대기 - CPU 소모 방지
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // ON 상태인데 카메라가 닫혀 있으면 재연결 시도
        if (!cap->isOpened()) {
            printf("[cam %d] trying to open (ON)...\n", camera_id);
            fflush(stdout);
            if (!cap->open(camera_id)) {
                fprintf(stderr, "[cam %d] open failed, retrying in 500ms\n",
                        camera_id);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;   // 재시도
            }
            printf("[cam %d] opened successfully (ON)\n", camera_id);
            fflush(stdout);
        }

        // ── 프레임 캡처 ────────────────────────────────────────────────────
        cv::Mat raw_frame;
        if (!cap->read(raw_frame) || raw_frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        // 전처리: resize → BGR→RGB 변환 (mutex 밖에서 수행 → 임계 구역 최소화)
        cv::Mat processed;
        cv::resize(raw_frame, processed, cv::Size(input_width, input_height));
        cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

        // input_queue 에 적재 (큐 깊이 초과 시 가장 오래된 프레임 드롭)
        {
            std::lock_guard<std::mutex> lock(input_mutex);
            if (input_queue.size() >= INPUT_QUEUE_MAX) {
                input_queue.pop();  // 오래된 프레임 드롭하여 실시간성 유지
            }
            input_queue.push({camera_id, processed.clone()});
        }
        // 추론 스레드가 condition_variable 로 대기 중이라면 깨워줌
        input_cv.notify_one();
    }
}

// ── Thread B (Consumer): 추론 및 파싱 ────────────────────────────────────────
//  condition_variable 로 input_queue 에 데이터가 생길 때까지 블록(Spinlock 방지)
//  → TFLite Invoke() → 박스 파싱 → output_queue.push
void inference_thread_func(tflite::Interpreter* interpreter,
                           int total_pixels) {
    while (running) {
        // ── input_queue 에서 FrameData 꺼내기 ──────────────────────────────
        FrameData frame;
        {
            std::unique_lock<std::mutex> lock(input_mutex);
            // 큐가 비어있는 동안 최대 100ms 대기 (Spinlock 방지)
            input_cv.wait_for(lock, std::chrono::milliseconds(100),
                              []{ return !input_queue.empty() || !running; });

            if (input_queue.empty()) continue;  // 타임아웃 또는 running=false

            frame = input_queue.front();
            input_queue.pop();
        }

        // ── TFLite 입력 텐서에 데이터 복사 (핵심 로직 변경 없음) ────────────
        TfLiteTensor* input_tensor_ptr =
            interpreter->tensor(interpreter->inputs()[0]);

        if (tpu_mode) {
            if (input_tensor_ptr->type == kTfLiteUInt8) {
                uint8_t* input_tensor =
                    interpreter->typed_input_tensor<uint8_t>(0);
                std::memcpy(input_tensor, frame.image.data, total_pixels);
            } else if (input_tensor_ptr->type == kTfLiteInt8) {
                int8_t* input_tensor =
                    interpreter->typed_input_tensor<int8_t>(0);
                for (int i = 0; i < total_pixels; ++i)
                    input_tensor[i] =
                        (int8_t)((int)frame.image.data[i] - 128);
            }
        } else {
            float* input_tensor =
                interpreter->typed_input_tensor<float>(0);
            for (int i = 0; i < total_pixels; ++i)
                input_tensor[i] = (float)frame.image.data[i] / 255.0f;
        }

        // ── TFLite Invoke() (핵심 로직 변경 없음) ────────────────────────────
        auto t0 = std::chrono::high_resolution_clock::now();
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "[cam " << frame.camera_id
                      << "] Inference failed!\n";
            continue;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> latency = t1 - t0;
        printf("\r[cam %d] Inference Latency: %.2f ms   ",
               frame.camera_id, latency.count());
        fflush(stdout);

        // ── 박스 파싱 ──────────────────────────────────────────────────────
        auto detections = parse_detections_thread_safe(
            interpreter,
            frame.image.cols,   // input_width
            frame.image.rows);  // input_height

        // 시각화용 BGR 이미지 생성 (RGB → BGR)
        cv::Mat display;
        cv::cvtColor(frame.image, display, cv::COLOR_RGB2BGR);

        // ── output_queue 에 적재 & 메인 스레드 알림 ───────────────────────
        {
            std::lock_guard<std::mutex> lock(output_mutex);
            if (output_queue.size() >= OUTPUT_QUEUE_MAX) {
                output_queue.pop();  // 오래된 결과 드롭
            }
            output_queue.push({frame.camera_id,
                               display.clone(),
                               detections});
        }
        output_cv.notify_one();
    }
}

// ── Main Thread: 시각화 및 GUI ────────────────────────────────────────────────
//  [output_mutex] output_queue.front() → pop()
//  → 박스 그리기 → cv::imshow("Yolo cam <id>") & cv::waitKey
int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "minimal <tflite model> <use tpu 0/1>\n");
        return 1;
    }
    const char* filename = argv[1];
    bool use_tpu = std::stoi(argv[2]);

    // ── [3] camera_active_flags 초기화 (기본값 true = ON) ────────────────────
    for (int i = 0; i < NUM_CAMERAS; ++i) {
        camera_active_flags[i] =
            std::unique_ptr<std::atomic<bool>>(new std::atomic<bool>(true));
    }

    // ── (1) 멀티 카메라 열기 (index 0, index 2) ───────────────────────────────
    const int cam_indices[NUM_CAMERAS] = {0, 2};

    cv::VideoCapture caps[NUM_CAMERAS];
    for (int i = 0; i < NUM_CAMERAS; ++i) {
        caps[i].open(cam_indices[i]);
        if (!caps[i].isOpened()) {
            fprintf(stderr, "Warning: cannot open camera index %d (will retry in thread)\n",
                    cam_indices[i]);
        } else {
            printf("Camera %d (index %d) opened.\n", i, cam_indices[i]);
        }
    }

    // ── (2) Load model ────────────────────────────────────────────────────────
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(filename);
    TFLITE_MINIMAL_CHECK(model != nullptr);

    // ── (3) Build interpreter ─────────────────────────────────────────────────
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    // ── (4) Setup for Edge TPU device (핵심 로직 변경 없음) ──────────────────
    if (use_tpu) {
        if (!(tpu_mode = setupEdgeTpu(interpreter.get()))) {
            std::cerr << "Falling back to CPU mode...\n";
        }
    }

    // ── (5) Allocate tensor buffers ───────────────────────────────────────────
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    printf("=== Pre-invoke Interpreter State ===\n");

    for (int i = 0; i < (int)interpreter->inputs().size(); i++) {
        TfLiteTensor* t = interpreter->tensor(interpreter->inputs()[i]);
        printf("Input [%d]: %s, Type: %d, Dims: ", i, t->name, t->type);
        for (int d = 0; d < t->dims->size; d++) printf("%d ", t->dims->data[d]);
        printf("\n");
    }
    for (int i = 0; i < (int)interpreter->outputs().size(); i++) {
        TfLiteTensor* t = interpreter->tensor(interpreter->outputs()[i]);
        printf("Output [%d]: %s, Type: %d, Dims: ", i, t->name, t->type);
        for (int d = 0; d < t->dims->size; d++) printf("%d ", t->dims->data[d]);
        printf("\n");
    }

    // 입력 텐서 크기 읽기
    TfLiteTensor* input_tensor_ptr =
        interpreter->tensor(interpreter->inputs()[0]);
    const int input_height = input_tensor_ptr->dims->data[1];
    const int input_width  = input_tensor_ptr->dims->data[2];
    const int channels     = input_tensor_ptr->dims->data[3];
    const int total_pixels = input_height * input_width * channels;

    // ── (5.4) Polling Thread 시작 ─────────────────────────────────────────────
    std::thread polling_thread(camera_status_polling_func);

    // ── (5.5) Thread A × NUM_CAMERAS 시작 (Producer) ─────────────────────────
    std::vector<std::thread> camera_threads;
    for (int i = 0; i < NUM_CAMERAS; ++i) {
        camera_threads.emplace_back(camera_thread_func,
                                    &caps[i], i,
                                    input_width, input_height);
    }

    // ── (5.6) Thread B 시작 (Consumer / 추론) ────────────────────────────────
    std::thread inference_thread(inference_thread_func,
                                  interpreter.get(),
                                  total_pixels);

    // ── Main Thread: 시각화 루프 ──────────────────────────────────────────────
    while (running) {
        DetectionResult result;
        {
            std::unique_lock<std::mutex> lock(output_mutex);
            // 결과가 올 때까지 최대 16ms(≈60fps) 대기 → Spinlock 방지
            output_cv.wait_for(lock, std::chrono::milliseconds(16),
                               []{ return !output_queue.empty() || !running; });

            if (output_queue.empty()) {
                // GUI 이벤트만 처리하고 다음 루프
                if (getenv("DISPLAY") != NULL) {
                    char key = cv::waitKey(1);
                    if (key == 'q') { running = false; break; }
                }
                continue;
            }
            result = output_queue.front();
            output_queue.pop();
        }

        // 박스 그리기
        yolo_output_visualize(result.display_image, result.detections);

        // 300×300 리사이즈 후 카메라별 창에 표시
        cv::Mat show_image;
        cv::resize(result.display_image, show_image, cv::Size(300, 300));
        std::string window_name = "Yolo cam " + std::to_string(result.camera_id);

        if (getenv("DISPLAY") != NULL) {
            try {
                cv::imshow(window_name, show_image);
            } catch (const cv::Exception& e) {
                // imshow 실패 시 무시
            }
            char key = cv::waitKey(1);
            if (key == 'q') {
                running = false;
                break;
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // ── (11) 정리 ─────────────────────────────────────────────────────────────
    running = false;
    input_cv.notify_all();   // 추론 스레드 대기 해제
    output_cv.notify_all();  // 메인 스레드 대기 해제

    if (polling_thread.joinable())   polling_thread.join();
    for (auto& t : camera_threads) {
        if (t.joinable()) t.join();
    }
    if (inference_thread.joinable()) inference_thread.join();

    for (int i = 0; i < NUM_CAMERAS; ++i) {
        caps[i].release();
    }
    cv::destroyAllWindows();
    return 0;
}
