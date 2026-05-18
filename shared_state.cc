#include "shared_state.h"

// ── 전역 변수 정의 ────────────────────────────────────────────────────────────
std::vector<std::unique_ptr<CameraSlot>> input_slots;
std::condition_variable                  input_cv;
std::mutex                               input_cv_mutex;

std::queue<DetectionResult> output_queue;
std::mutex                  output_mutex;
std::condition_variable     output_cv;

std::vector<std::unique_ptr<std::atomic<bool>>> camera_active_flags;

std::atomic<bool> running{true};
bool              tpu_mode = false;

cv::Mat latest_display_frame;
std::mutex latest_frame_mutex;
