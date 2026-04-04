#pragma once
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

// Thread-safe output parsing (using TFLite PostProcess Op)
inline std::vector<Detection> parse_detections_thread_safe(tflite::Interpreter* interpreter, int img_width, int img_height) {
    std::vector<Detection> detections;
    
    // Check if enough outputs exist
    if (interpreter->outputs().size() < 4) return detections;
    
    // SSD-style models (regardless of name) often use the following 4 output tensors:
    // Output 0: Locations, Output 1: Classes, Output 2: Scores, Output 3: Count
    float* boxes   = interpreter->tensor(interpreter->outputs()[0])->data.f;
    float* classes = interpreter->tensor(interpreter->outputs()[1])->data.f;
    float* scores  = interpreter->tensor(interpreter->outputs()[2])->data.f;
    float  count   = interpreter->tensor(interpreter->outputs()[3])->data.f[0];

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
        sprintf(label, "%s, Score: %.2f", class_name.c_str(), det.score);
        
        cv::putText(image, label, cv::Point(det.box.x, det.box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 3);
        cv::putText(image, label, cv::Point(det.box.x, det.box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}
