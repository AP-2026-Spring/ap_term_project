#include "yolo_parser.h"
#include <map>
#include <string>
#include <vector>
#include "tensorflow/lite/interpreter.h"

// Thread-safe detection result structure
struct Detection {
    int class_id;
    float score;
    cv::Rect box;
};

// Helper to get float data from a tensor (handles dequantization)
inline std::vector<float> get_tensor_data_as_float(TfLiteTensor* tensor) {
    int size = 1;
    for (int i = 0; i < tensor->dims->size; i++) size *= tensor->dims->data[i];
    
    std::vector<float> data(size);
    if (tensor->type == kTfLiteFloat32) {
        float* raw = (float*)tensor->data.data;
        std::copy(raw, raw + size, data.begin());
    } else if (tensor->type == kTfLiteUInt8) {
        uint8_t* raw = (uint8_t*)tensor->data.data;
        for (int i = 0; i < size; i++) {
            data[i] = (static_cast<float>(raw[i]) - tensor->params.zero_point) * tensor->params.scale;
        }
    } else if (tensor->type == kTfLiteInt8) {
        int8_t* raw = (int8_t*)tensor->data.data;
        for (int i = 0; i < size; i++) {
            data[i] = (static_cast<float>(raw[i]) - tensor->params.zero_point) * tensor->params.scale;
        }
    }
    return data;
}

// Thread-safe output parsing (using yolo_parser.h logic)
inline std::vector<Detection> yolo_parse_detections(TfLiteTensor* cls_tensor, TfLiteTensor* loc_tensor, int img_width, int img_height){
  static yolo::YOLO_Parser parser;
  
  // Clear static states
  yolo::YOLO_Parser::real_bbox_cls_index_vector.clear();
  yolo::YOLO_Parser::real_bbox_loc_vector.clear();
  yolo::YOLO_Parser::result_boxes.clear();
  
  // (임시) YOLO_Parser의 로직이 내부적으로 TfLiteTensor의 data.data를 직접 float*로 캐스팅하여 사용하므로, 
  // 만약 텐서가 양자화(UInt8/Int8)되어 있다면 강제로 Float 텐서 구조를 흉내내거나 로직을 우회해야 합니다.
  // 여기서는 안전하게 YOLO_Parser를 거치지 않고 직접 파싱하는 것이 더 안정적일 수 있으나, 
  // 일단 YOLO_Parser가 기대하는 float* 데이터를 위해 임시 버퍼를 사용하는 방식을 시도합니다.
  
  // 하지만 YOLO_Parser 코드를 보면 TfLiteTensor*를 인자로 받아 내부에서 다시 캐스팅하므로, 
  // 텐서 데이터 자체를 Float으로 변환한 "가짜" 텐서를 넘기는 것은 어렵습니다.
  
  // 따라서, 텐서 타입이 Float인지 먼저 확인하고 아니라면 경고를 출력합니다.
  if (cls_tensor->type != kTfLiteFloat32 || loc_tensor->type != kTfLiteFloat32) {
      printf("\033[1;31mWarning: Model outputs are quantized (Type %d, %d). YOLO_Parser may crash!\033[0m\n", 
             cls_tensor->type, loc_tensor->type);
  }

  std::vector<int> real_bbox_index_vector;
  std::vector<std::vector<float>> cls_vector;
  std::vector<std::vector<int>> loc_vector;
  
  // Make classification vector (내부에서 (float*)로 캐스팅함)
  parser.make_real_bbox_cls_vector(cls_tensor, real_bbox_index_vector, cls_vector);
  
  std::vector<int> cls_index_vector = parser.get_cls_index(cls_vector); 
  
  // Make localization vector (내부에서 (float*)로 캐스팅함)
  parser.make_real_bbox_loc_vector(loc_tensor, real_bbox_index_vector, loc_vector);
  
  // NMS
  float iou_threshold = 0.5;
  parser.PerformNMSUsingResults(real_bbox_index_vector, cls_vector, loc_vector, iou_threshold, cls_index_vector);
  
  // Convert parser results to Detection struct
  std::vector<Detection> detections;
  for (const auto& bbox : yolo::YOLO_Parser::result_boxes) {
      float scale_x = (float)img_width / IMG_size;
      float scale_y = (float)img_height / IMG_size;
      detections.push_back({
          bbox.class_id, 
          bbox.score, 
          cv::Rect(
              static_cast<int>(bbox.left * scale_x),
              static_cast<int>(bbox.top * scale_y),
              static_cast<int>((bbox.right - bbox.left) * scale_x),
              static_cast<int>((bbox.bottom - bbox.top) * scale_y)
          )
      });
  }
  return detections;
}

// Current COCO label dictionary
inline std::map<int, std::string> get_coco_label_dict() {
    return {
        {0, "person"},     {1, "bicycle"},   {2, "car"},          {3, "motorbike"},
        {4, "aeroplane"},  {5, "bus"},       {6, "train"},        {7, "truck"},
        {8, "boat"},       {9, "traffic_light"}, {10, "fire_hydrant"}, {11, "stop_sign"},
        {12, "parking_meter"}, {13, "bench"}, {14, "bird"},       {15, "cat"},
        {16, "dog"},       {17, "horse"},    {18, "sheep"},       {19, "cow"},
        {20, "elephant"},  {21, "bear"},     {22, "zebra"},       {23, "giraffe"},
        {24, "backpack"},  {25, "umbrella"}, {26, "handbag"},     {27, "tie"},
        {28, "suitcase"},  {29, "frisbee"},  {30, "skis"},        {31, "snowboard"},
        {32, "sports_ball"}, {33, "kite"},   {34, "baseball_bat"}, {35, "baseball_glove"},
        {36, "skateboard"}, {37, "surfboard"}, {38, "tennis_racket"}, {39, "bottle"},
        {40, "wine_glass"}, {41, "cup"},     {42, "fork"},        {43, "knife"},
        {44, "spoon"},     {45, "bowl"},    {46, "banana"},      {47, "apple"},
        {48, "sandwich"},  {49, "orange"},  {50, "broccoli"},    {51, "carrot"},
        {52, "hot_dog"},   {53, "pizza"},   {54, "donut"},       {55, "cake"},
        {56, "chair"},     {57, "sofa"},    {58, "potted_plant"}, {59, "bed"},
        {60, "dining_table"}, {61, "toilet"}, {62, "tvmonitor"}, {63, "laptop"},
        {64, "mouse"},     {65, "remote"},  {66, "keyboard"},    {67, "cell_phone"},
        {68, "microwave"}, {69, "oven"},    {70, "toaster"},     {71, "sink"},
        {72, "refrigerator"}, {73, "book"}, {74, "clock"},       {75, "vase"},
        {76, "scissors"},  {77, "teddy_bear"}, {78, "hair_drier"}, {79, "toothbrush"}
    };
}

// Thread-safe visualization
inline void yolo_output_visualize(cv::Mat& image, const std::vector<Detection>& detections) {
    static std::map<int, std::string> labelDict = get_coco_label_dict();
    for (const auto& det : detections) {
        cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);

        char label[256];
        std::string class_name = labelDict.count(det.class_id) ? labelDict[det.class_id] : std::to_string(det.class_id);
        sprintf(label, "%s: %.2f", class_name.c_str(), det.score);
        
        cv::putText(image, label, cv::Point(det.box.x, det.box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 3);
        cv::putText(image, label, cv::Point(det.box.x, det.box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}
