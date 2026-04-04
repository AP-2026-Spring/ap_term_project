#include "shared_state.h"

// ── 전역 변수 정의 ────────────────────────────────────────────────────────────
std::queue<FrameData>       input_queue;
std::mutex                  input_mutex;
std::condition_variable     input_cv;

std::queue<DetectionResult> output_queue;
std::mutex                  output_mutex;
std::condition_variable     output_cv;

std::vector<std::unique_ptr<std::atomic<bool>>> camera_active_flags;

std::atomic<bool> running{true};
bool              tpu_mode = false;
