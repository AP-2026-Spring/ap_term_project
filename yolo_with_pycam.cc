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

using namespace std;

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

// (새로운) 모델 출력 파싱 및 바운딩 박스 그리기 함수
void parse_and_visualize_postprocess(tflite::Interpreter* interpreter, cv::Mat& image) {
    // 1. 모델에서 4개의 출력 텐서 데이터 가져오기
    float* boxes   = interpreter->tensor(interpreter->outputs()[0])->data.f; // [1, 20, 4] 박스 좌표
    float* classes = interpreter->tensor(interpreter->outputs()[1])->data.f; // [1, 20] 클래스 ID
    float* scores  = interpreter->tensor(interpreter->outputs()[2])->data.f; // [1, 20] 신뢰도 점수
    float  count   = interpreter->tensor(interpreter->outputs()[3])->data.f[0]; // [1] 감지된 객체 수

// [디버깅] 모델이 도대체 무슨 값을 뱉고 있는지 터미널에 출력!
    printf("Detected Count: %f\n", count);
    if (count > 0) {
        printf("Top 1 Score: %f (Class: %d)\n", scores[0], (int)classes[0]);
        printf("Top 1 Box: [ymin:%.2f, xmin:%.2f, ymax:%.2f, xmax:%.2f]\n", 
               boxes[0], boxes[1], boxes[2], boxes[3]);
    }

    int img_width = image.cols;
    int img_height = image.rows;

    // 2. 감지된 객체 개수만큼 반복하며 화면에 그리기
    for (int i = 0; i < (int)count; ++i) {
        // 신뢰도 50% 이상인 객체만 표시 (필요시 0.3 등으로 조절)
        if (scores[i] >= 0.5f) {
            // 좌표 비율(0.0~1.0)을 실제 이미지 픽셀 크기로 변환
            // 주의: TFLite PostProcess는 ymin, xmin, ymax, xmax 순서로 뱉습니다.
            int ymin = (int)(boxes[i * 4 + 0] * img_height);
            int xmin = (int)(boxes[i * 4 + 1] * img_width);
            int ymax = (int)(boxes[i * 4 + 2] * img_height);
            int xmax = (int)(boxes[i * 4 + 3] * img_width);

            // 박스가 화면 밖으로 튀어나가지 않게 안전장치(Clamping)
            ymin = std::max(0, ymin);
            xmin = std::max(0, xmin);
            ymax = std::min(img_height - 1, ymax);
            xmax = std::min(img_width - 1, xmax);

            // 초록색 바운딩 박스 그리기
            cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 2);

            // 텍스트 (클래스 ID와 확률) 작성
            char label[256];
            sprintf(label, "ID: %d, Score: %.2f", (int)classes[i], scores[i]);
            
            // 글씨 배경에 살짝 검은색 그림자를 주면 더 잘 보입니다 (선택 사항)
            cv::putText(image, label, cv::Point(xmin, ymin - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 3); // 그림자
            cv::putText(image, label, cv::Point(xmin, ymin - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1); // 실제 글씨
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

  while (true) {
    // (6) Load image from Pycam
    cv::Mat image;
//     camera.grab();
//     camera.retrieve(image);
    cap >> image;
    cv::imshow("Yolo example with Pycam", image);
    if (image.empty()) {
      cerr << "Error capturing image" << endl;
      break;
    }
    vector<cv::Mat> input;
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(300, 300));
    input.push_back(image);

    TfLiteTensor* input_tensor_ptr = interpreter->tensor(interpreter->inputs()[0]);
    int input_width = input_tensor_ptr->dims->data[2];
    int input_height = input_tensor_ptr->dims->data[1];
    int channels = input_tensor_ptr->dims->data[3];
    int total_pixels = input_width * input_height * channels;

    // (7) Push image to input tensor
    if (tpu_mode) {
      if (input_tensor_ptr->type == kTfLiteUInt8) {
        uint8_t* input_tensor = interpreter->typed_input_tensor<uint8_t>(0);
        std::memcpy(input_tensor, image.data, total_pixels);
      } else if (input_tensor_ptr->type == kTfLiteInt8) {
        int8_t* input_tensor = interpreter->typed_input_tensor<int8_t>(0);
        for (int i = 0; i < total_pixels; ++i)
            input_tensor[i] = (int8_t)((int)image.data[i] - 128); 
      }
    } else {
      float* input_tensor = interpreter->typed_input_tensor<float>(0);
      for (int i = 0; i < 416 * 416 * 3; ++i)
      input_tensor[i] = (float) image.data[i] / 255.0f;
    }

    // (8) Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    printf("\n\n=== Post-invoke Interpreter State ===\n");

printf("\n=== Output Tensor Check ===\n");
int num_outputs = interpreter->outputs().size();
printf("Number of output tensors: %d\n", num_outputs);

for (int i = 0; i < num_outputs; i++) {
    TfLiteTensor* out_tensor = interpreter->tensor(interpreter->outputs()[i]);
    printf("Output %d - Name: %s, Dims: [", i, out_tensor->name);
    for (int d = 0; d < out_tensor->dims->size; d++) {
        printf("%d ", out_tensor->dims->data[d]);
    }
    printf("]\n");
}
printf("===========================\n\n");

    // (9) Output parsing
    parse_and_visualize_postprocess(interpreter.get(), image); 

    // (10) Output visualize
    yolo_output_visualize(image);

    char key = cv::waitKey(1);
    if (key == 'q') {
        break;
    }
  }
  // (11) release
  cap.release();
  cv::destroyAllWindows();
//   camera.release();
// 	cv::destroyAllWindows();
  return 0;
}

  
