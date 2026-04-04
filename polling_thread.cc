#include "polling_thread.h"
#include "shared_state.h"
#include <cstdio>
#include <string>
#include <sstream>
#include <fstream>
#include <thread>
#include <chrono>

// ── Polling Thread: 공유 파일 폴링으로 카메라 On/Off 제어 ────────────────────
//  1초마다 CAM_CMD_PATH 를 읽어 camera_active_flags 갱신
//  파일 없음 → 조용히 skip
//  파싱 실패·예외 → 프로그램 중단 없이 다음 주기 재시도
void camera_status_polling_func() {
    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        try {
            std::ifstream ifs(CAM_CMD_PATH);
            if (!ifs.is_open()) continue;   // 파일 없으면 조용히 넘어감

            std::string line;
            while (std::getline(ifs, line)) {
                if (line.empty()) continue;

                // 포맷 파싱: "cam_id:state" (예: "0:1", "1:0")
                std::istringstream ss(line);
                std::string token_id, token_state;
                if (!std::getline(ss, token_id,    ':')) continue;
                if (!std::getline(ss, token_state, ':')) continue;

                int cam_id = -1, state = -1;
                try {
                    cam_id = std::stoi(token_id);
                    state  = std::stoi(token_state);
                } catch (...) {
                    continue;   // 정수 변환 실패 → 해당 줄 무시
                }

                if (cam_id < 0 || cam_id >= (int)camera_active_flags.size()) continue;
                if (state != 0 && state != 1) continue;

                const bool new_active = (state == 1);
                if (camera_active_flags[cam_id]->load() != new_active) {
                    camera_active_flags[cam_id]->store(new_active);
                    printf("[polling] camera %d → %s\n",
                           cam_id, new_active ? "ON" : "OFF");
                    fflush(stdout);
                }
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "[polling] file I/O exception: %s\n", e.what());
        } catch (...) {
            fprintf(stderr, "[polling] unknown file I/O exception\n");
        }
    }
}
