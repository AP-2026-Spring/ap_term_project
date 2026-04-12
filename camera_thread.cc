#include "camera_thread.h"
#include "shared_state.h"
#include <cstdio>
#include <thread>
#include <chrono>

// ── Thread A (Producer): 카메라 캡처 & 전처리 & 하드웨어 On/Off 제어 ──────────
//  · flag=false & cap.isOpened()  → cap.release()  (하드웨어 점유 해제)
//  · flag=true  & !cap.isOpened() → cap.open()     (하드웨어 재연결, 최대 5회)
//  · Off 상태 동안 100ms Idle (CPU 소모 방지)
//  · 캡처 성공 시: resize → cvtColor(BGR→RGB) → input_queue.push → input_cv.notify
void camera_thread_func(cv::VideoCapture* cap, int camera_id,
                        int input_width, int input_height) {
    int retry_count = 0;  // 이 스레드 전용 재연결 시도 횟수 (thread-safe)

    while (running) {
        const bool is_active = camera_active_flags[camera_id]->load();

        // ── 하드웨어 On/Off 전환 처리 ──────────────────────────────────────
        if (!is_active) {
            if (cap->isOpened()) {
                cap->release();
                printf("[cam %d] released (OFF)\n", camera_id);
                fflush(stdout);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // ON 상태인데 카메라가 닫혀 있으면 재연결 시도 (최대 5회)
        if (!cap->isOpened()) {
            printf("[cam %d] trying to open (ON)... [%d/5]\n",
                   camera_id, retry_count + 1);
            fflush(stdout);

            if (!cap->open(camera_id)) {
                ++retry_count;
                fprintf(stderr, "[cam %d] open failed (retry %d/5)\n",
                        camera_id, retry_count);
                if (retry_count >= 5) {
                    fprintf(stderr,
                            "[cam %d] max retries reached. auto-disabling.\n",
                            camera_id);
                    camera_active_flags[camera_id]->store(false);
                    retry_count = 0;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }
            retry_count = 0;
            printf("[cam %d] opened successfully (ON)\n", camera_id);
            fflush(stdout);
        }

        // ── 프레임 캡처 ────────────────────────────────────────────────────
        cv::Mat raw_frame;
        if (!cap->read(raw_frame) || raw_frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        // 전처리: resize → BGR→RGB 변환 (mutex 밖에서 수행 → 임계 구역 최소화)
        cv::Mat processed;
        cv::resize(raw_frame, processed, cv::Size(input_width, input_height));
        cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

        // input_slots[camera_id] 업데이트 (Latest-wins: 기존 프레임 덮어쓰기)
        {
            std::lock_guard<std::mutex> lock(input_slots[camera_id]->mtx);
            input_slots[camera_id]->image = processed.clone();
            input_slots[camera_id]->ready = true;
        }
        
        // 추론 스레드에 새 프레임 도착 알림
        input_cv.notify_all();
    }
}
