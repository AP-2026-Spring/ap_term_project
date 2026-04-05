#pragma once
#include <map>
#include <string>
#include <vector>
#include "tensorflow/lite/interpreter.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Thread-safe detection result structure
struct Detection {
    int class_id;
    float score;
    cv::Rect box;
};

// Thread-safe output parsing (using TFLite PostProcess Op)
inline std::vector<Detection> parse_detections_thread_safe(tflite::Interpreter* interpreter, int img_width, int img_height) {
    std::vector<Detection> detections;
    
    // Check if the model is YOLOv8 INT8 with a single output tensor
    if (interpreter->outputs().size() == 1) {
        TfLiteTensor* output = interpreter->tensor(interpreter->outputs()[0]);
        if (output->type == kTfLiteInt8 && output->dims->size == 3) {
            int8_t* data = output->data.int8;
            float scale = output->params.scale;
            int zero_point = output->params.zero_point;
            
            int num_channels = output->dims->data[1]; // e.g., 6 (4 bbox + 2 classes)
            int num_anchors = output->dims->data[2];  // e.g., 2100
            
            std::vector<cv::Rect> boxes;
            std::vector<float> scores;
            std::vector<int> class_ids;
            
            static bool debug_printed = false;
            if (!debug_printed) {
                printf("\n--- QUANTIZATION DEBUG ---\n");
                printf("scale: %f, zero_point: %d\n", scale, zero_point);
                float max_s = -1000.0f;
                for (int idx = 0; idx < num_anchors; ++idx) {
                    float s0 = (data[4 * num_anchors + idx] - zero_point) * scale;
                    float s1 = (data[5 * num_anchors + idx] - zero_point) * scale;
                    if (s0 > max_s) max_s = s0;
                    if (s1 > max_s) max_s = s1;
                }
                printf("Global Max Score across all anchors: %f\n", max_s);
                printf("First Anchor Raw: cx=%d, cy=%d, w=%d, h=%d, s0=%d, s1=%d\n", 
                       data[0*num_anchors], data[1*num_anchors], data[2*num_anchors], data[3*num_anchors], data[4*num_anchors], data[5*num_anchors]);
                printf("First Anchor Float: cx=%f, cy=%f, w=%f, h=%f, s0=%f, s1=%f\n",
                       (data[0*num_anchors]-zero_point)*scale, (data[1*num_anchors]-zero_point)*scale, 
                       (data[2*num_anchors]-zero_point)*scale, (data[3*num_anchors]-zero_point)*scale,
                       (data[4*num_anchors]-zero_point)*scale, (data[5*num_anchors]-zero_point)*scale);
                printf("-------------------------\n");
                debug_printed = true;
            }
            
            for (int i = 0; i < num_anchors; ++i) {
                float max_score = -1.0f;
                int max_class = -1;
                // Find class with the maximum confidence score
                for (int c = 4; c < num_channels; ++c) {
                    float score = (data[c * num_anchors + i] - zero_point) * scale;
                    if (score > max_score) {
                        max_score = score;
                        max_class = c - 4;
                    }
                }
                
                if (max_score >= 0.25f) {
                    // Extract absolute box coordinates from the first 4 channels
                    float cx = (data[0 * num_anchors + i] - zero_point) * scale;
                    float cy = (data[1 * num_anchors + i] - zero_point) * scale;
                    float w = (data[2 * num_anchors + i] - zero_point) * scale;
                    float h = (data[3 * num_anchors + i] - zero_point) * scale;
                    
                    float xmin = cx - w / 2.0f;
                    float ymin = cy - h / 2.0f;
                    
                    int xmin_i = std::max(0, (int)xmin);
                    int ymin_i = std::max(0, (int)ymin);
                    int xmax_i = std::min(img_width - 1, (int)(cx + w / 2.0f));
                    int ymax_i = std::min(img_height - 1, (int)(cy + h / 2.0f));
                    
                    if (xmax_i > xmin_i && ymax_i > ymin_i) {
                        boxes.push_back(cv::Rect(xmin_i, ymin_i, xmax_i - xmin_i, ymax_i - ymin_i));
                        scores.push_back(max_score);
                        class_ids.push_back(max_class);
                    }
                }
            }
            
            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, scores, 0.25f, 0.45f, indices);
            
            for (int idx : indices) {
                detections.push_back({class_ids[idx], scores[idx], boxes[idx]});
            }
            return detections;
        }
    }
    
    // Fallback for SSD-style models (4 output tensors)
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
        {0, "cockroach"},     {1, "mouse"},   {2, "car"},          {3, "motorbike"},
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
