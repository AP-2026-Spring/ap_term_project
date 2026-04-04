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
#include <vector>
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
// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: ./minimal_yolo <tflite model>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

using namespace std;

// Shared data for threading
struct Detection {
    int class_id;
    float score;
    cv::Rect box;
};

std::mutex frame_mutex;
cv::Mat shared_frame;
std::atomic<bool> new_frame_available{false};

std::mutex results_mutex;
std::vector<Detection> shared_results;

std::atomic<bool> running{true};

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

bool tpu_mode = 0;

bool setupEdgeTpu(tflite::Interpreter* interpreter) {
    size_t num_devices;
    
    // 1. 연결된 TPU 장치 목록 확인
    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

    if (num_devices == 0) {
        std::cerr << "Error: No Edge TPU devices found! Check your connection." << std::endl;
        return false;
    }

    std::cout << "TPU MODE: Found " << num_devices << " device(s)." << std::endl;

    // 2. 첫 번째 장치 선택 및 델리게이트 생성
    const auto& device = devices.get()[0];
    auto* delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);

    // 3. Interpreter의 그래프를 TPU용으로 수정
    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
        std::cerr << "Error: Failed to modify graph with Edge TPU delegate!" << std::endl;
        return false;
    }

    std::cout << "Successfully applied Edge TPU delegate to the interpreter." << std::endl;
    return true;
}

// (새로운) 모델 출력 파싱 함수 (데이터만 추출)
std::vector<Detection> parse_detections(tflite::Interpreter* interpreter, int img_width, int img_height) {
    std::vector<Detection> detections;
    
    // 1. 모델에서 4개의 출력 텐서 데이터 가져오기
    float* boxes   = interpreter->tensor(interpreter->outputs()[0])->data.f; // [1, 20, 4] 박스 좌표
    float* classes = interpreter->tensor(interpreter->outputs()[1])->data.f; // [1, 20] 클래스 ID
    float* scores  = interpreter->tensor(interpreter->outputs()[2])->data.f; // [1, 20] 신뢰도 점수
    float  count   = interpreter->tensor(interpreter->outputs()[3])->data.f[0]; // [1] 감지된 객체 수

    // 2. 감지된 객체 개수만큼 반복하며 데이터 추출
    for (int i = 0; i < (int)count; ++i) {
        if (scores[i] >= 0.5f) {
            int ymin = (int)(boxes[i * 4 + 0] * img_height);
            int xmin = (int)(boxes[i * 4 + 1] * img_width);
            int ymax = (int)(boxes[i * 4 + 2] * img_height);
            int xmax = (int)(boxes[i * 4 + 3] * img_width);

            ymin = std::max(0, ymin);
            xmin = std::max(0, xmin);
            ymax = std::min(img_height - 1, ymax);
            xmax = std::min(img_width - 1, xmax);

            detections.push_back({(int)classes[i], scores[i], cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin)});
        }
    }
    return detections;
}

// 추출된 결과를 화면에 그리는 함수 (Main Thread에서 호출)
void draw_detections(cv::Mat& image, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        // 초록색 바운딩 박스 그리기
        cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);

        // 텍스트 (클래스 ID와 확률) 작성
        char label[256];
        sprintf(label, "ID: %d, Score: %.2f", det.class_id, det.score);
        
        cv::putText(image, label, cv::Point(det.box.x, det.box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 3); // 그림자
        cv::putText(image, label, cv::Point(det.box.x, det.box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1); // 실제 글씨
    }
}

void inference_thread_func(tflite::Interpreter* interpreter, bool tpu_mode) {
    while (running) {
        if (!new_frame_available) {
            std::this_thread::yield();
            continue;
        }

        cv::Mat local_frame;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            local_frame = shared_frame.clone();
            new_frame_available = false;
        }

        if (local_frame.empty()) continue;

        // Preprocessing
        cv::Mat processed_image;
        cv::cvtColor(local_frame, processed_image, cv::COLOR_BGR2RGB);
        cv::resize(processed_image, processed_image, cv::Size(300, 300));

        TfLiteTensor* input_tensor_ptr = interpreter->tensor(interpreter->inputs()[0]);
        int input_width = input_tensor_ptr->dims->data[2];
        int input_height = input_tensor_ptr->dims->data[1];
        int channels = input_tensor_ptr->dims->data[3];
        int total_pixels = input_width * input_height * channels;

        if (tpu_mode) {
            if (input_tensor_ptr->type == kTfLiteUInt8) {
                uint8_t* input_tensor = interpreter->typed_input_tensor<uint8_t>(0);
                std::memcpy(input_tensor, processed_image.data, total_pixels);
            } else if (input_tensor_ptr->type == kTfLiteInt8) {
                int8_t* input_tensor = interpreter->typed_input_tensor<int8_t>(0);
                for (int i = 0; i < total_pixels; ++i)
                    input_tensor[i] = (int8_t)((int)processed_image.data[i] - 128); 
            }
        } else {
            float* input_tensor = interpreter->typed_input_tensor<float>(0);
            for (int i = 0; i < total_pixels; ++i)
                input_tensor[i] = (float)processed_image.data[i] / 255.0f;
        }

        // Run inference
        if (interpreter->Invoke() == kTfLiteOk) {
            // 헤더의 스레드 안전 파싱 함수 사용
            auto detections = parse_detections_thread_safe(interpreter, local_frame.cols, local_frame.rows);
            
            std::lock_guard<std::mutex> lock(results_mutex);
            shared_results = std::move(detections);
        }
    }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "minimal <tflite model> <use tpu 0/1> \n");
    return 1;
  }
  const char* filename = argv[1];
  bool use_tpu = std::stoi(argv[2]);
  
  // (1) Pycam setting
//   raspicam::RaspiCam_Cv camera;
//   camera.set(cv::CAP_PROP_FORMAT, CV_8UC3);
//   camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
//   camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
//   if (!camera.open()) {
//     cerr << "Error opening the camera" << endl;
//     return 1;
//   }
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    cerr << "Error opening the camera" << endl;
    return 1;
  }

  // (2) Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // (3) Build interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // (4) Setup for Edge TPU device.
  if(use_tpu){
		if (!(tpu_mode=setupEdgeTpu(interpreter.get()))) {
			std::cerr << "Falling back to CPU mode...\n";
		}
	}

  // (5) Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");

  // (5.5) Start Inference Thread
  std::thread inference_thread(inference_thread_func, interpreter.get(), tpu_mode);

  while (true) {
    // (6) Load image from Camera
    cv::Mat image;
    cap >> image;
    if (image.empty()) {
      cerr << "Error capturing image" << endl;
      break;
    }

    // Update shared frame for inference thread
    {
        std::lock_guard<std::mutex> lock(frame_mutex);
        shared_frame = image.clone(); // Clone to avoid modification while UI thread uses it
        new_frame_available = true;
    }

    // (7) Get latest results and draw
    std::vector<Detection> results_to_draw;
    {
        std::lock_guard<std::mutex> lock(results_mutex);
        results_to_draw = shared_results;
    }

    // 헤더에서 제공되는 시각화 함수 사용 (라벨 포함)
    yolo_output_visualize(image, results_to_draw);

    cv::imshow("Yolo example with Pycam", image);

    char key = cv::waitKey(1);
    if (key == 'q') {
        running = false;
        break;
    }
  }

  // (11) release
  running = false;
  if (inference_thread.joinable()) {
      inference_thread.join();
  }
  // (11) release
  cap.release();
  cv::destroyAllWindows();
//   camera.release();
// 	cv::destroyAllWindows();
  return 0;
}

  
