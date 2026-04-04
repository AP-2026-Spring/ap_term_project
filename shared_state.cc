#include "shared_state.h"
#include "yolo_parser.h"   // yolo::YOLO_Parser static member definitions

// ── YOLO_Parser 정적 멤버 정의 ────────────────────────────────────────────────
std::vector<std::vector<float>> yolo::YOLO_Parser::real_bbox_cls_vector;
std::vector<int>                yolo::YOLO_Parser::real_bbox_cls_index_vector;
std::vector<std::vector<int>>   yolo::YOLO_Parser::real_bbox_loc_vector;
std::vector<yolo::YOLO_Parser::BoundingBox> yolo::YOLO_Parser::result_boxes;

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
