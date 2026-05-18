#pragma once
#include <queue>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <vector>
#include <memory>
#include "opencv2/opencv.hpp"
#include "yolo_with_pycam.h"   // Detection struct, parse_detections_thread_safe, yolo_output_visualize

// ── 큐 최대 깊이 ──────────────────────────────────────────────────────────────
static constexpr size_t INPUT_QUEUE_MAX  = 4;
static constexpr size_t OUTPUT_QUEUE_MAX = 4;

// ── 폴링 파일 경로 (RAM 디스크 - SD 카드 수명 보호) ──────────────────────────
static constexpr const char* CAM_CMD_PATH = "/dev/shm/camera_cmd.txt";

// ── 내부 통신용 구조체 ──────────────────────────────────────────────────────
struct FrameData {
    int     camera_id;
    cv::Mat image;
};

// ── Producer → Consumer 전달 구조체 (Per-Camera Slot) ───────────────────────
struct CameraSlot {
    cv::Mat image;
    bool    ready = false;
    std::mutex mtx;
};

// ── Consumer → Main Thread 전달 구조체 ───────────────────────────────────────
// display_image: 시각화용 BGR 이미지
// detections:    파싱된 박스 목록
struct DetectionResult {
    int                    camera_id;
    cv::Mat                display_image;
    std::vector<Detection> detections;
};

// ── 전역 변수 extern 선언 (정의는 shared_state.cc) ───────────────────────────
extern std::vector<std::unique_ptr<CameraSlot>> input_slots;
extern std::condition_variable                  input_cv; 
extern std::mutex                               input_cv_mutex; // CV용 별도 뮤텍스

extern std::queue<DetectionResult> output_queue;
extern std::mutex                  output_mutex;
extern std::condition_variable     output_cv;

// camera_active_flags[i]: i번 카메라 On(true)/Off(false) 상태
// atomic<bool>은 복사 불가 → unique_ptr로 래핑하여 vector에 보관
extern std::vector<std::unique_ptr<std::atomic<bool>>> camera_active_flags;

extern std::atomic<bool> running;
extern bool              tpu_mode;

// MJPEG 스트리밍 서버용 글로벌 프레임 저장소
extern cv::Mat latest_display_frame;
extern std::mutex latest_frame_mutex;
