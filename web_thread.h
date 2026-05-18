#pragma once
#include <thread>

// MJPEG 스트리밍 서버 스레드
// 8080 포트에서 클라이언트 접속을 대기하고, latest_display_frame을 전송
void mjpeg_server_func();

// 웹소켓 클라이언트 스레드
// 백엔드(192.168.1.10:8081/detection)에 연결하여 카메라 ON/OFF 명령을 수신
void websocket_client_func();
