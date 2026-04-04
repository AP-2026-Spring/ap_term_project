#pragma once

// ── Polling Thread: 공유 파일 폴링으로 카메라 On/Off 제어 ────────────────────
// 1초 주기로 /dev/shm/camera_cmd.txt 를 읽어 camera_active_flags 갱신
// 파일 포맷: "0:1"(0번 ON), "1:0"(1번 OFF) — 여러 줄 가능
void camera_status_polling_func();
