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

// Thread-safe output parsing (using yolo_parser.h logic)
inline std::vector<Detection> yolo_parse_detections(TfLiteTensor* cls_tensor, TfLiteTensor* loc_tensor, int img_width, int img_height){
  static yolo::YOLO_Parser parser; // Thread-safe as result boxes are returned
  
  std::vector<int> real_bbox_index_vector;
  std::vector<std::vector<float>> cls_vector;
  std::vector<std::vector<int>> loc_vector;
  
  // Make classification vector
  parser.make_real_bbox_cls_vector(cls_tensor, real_bbox_index_vector, cls_vector);
  
  std::vector<int> cls_index_vector = parser.get_cls_index(cls_vector); 
  
  // Make localization vector
  parser.make_real_bbox_loc_vector(loc_tensor, real_bbox_index_vector, loc_vector);
  
  // NMS
  float iou_threshold = 0.5;
  parser.PerformNMSUsingResults(real_bbox_index_vector, cls_vector, loc_vector, iou_threshold, cls_index_vector);
  
  // Convert parser results (result_boxes) to our Detection struct
  std::vector<Detection> detections;
  for (const auto& bbox : yolo::YOLO_Parser::result_boxes) {
      // bbox left/top/right/bottom typically in IMG_size (416) scale
      float scale_x = (float)img_width / IMG_size;
      float scale_y = (float)img_height / IMG_size;
      
      int x = static_cast<int>(bbox.left * scale_x);
      int y = static_cast<int>(bbox.top * scale_y);
      int w = static_cast<int>((bbox.right - bbox.left) * scale_x);
      int h = static_cast<int>((bbox.bottom - bbox.top) * scale_y);
      
      detections.push_back({bbox.class_id, bbox.score, cv::Rect(x, y, w, h)});
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
