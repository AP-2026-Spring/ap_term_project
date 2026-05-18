#include "web_thread.h"
#include "shared_state.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <unistd.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>

// ── 1. MJPEG 스트리밍 서버 ──────────────────────────────────────────
void mjpeg_server_func() {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) return;

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    // Non-blocking 설정 (종료 시 accept에서 무한 대기 방지)
    fcntl(server_fd, F_SETFL, O_NONBLOCK);

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(8080);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "[MJPEG] Bind failed\n";
        close(server_fd);
        return;
    }

    if (listen(server_fd, 3) < 0) {
        std::cerr << "[MJPEG] Listen failed\n";
        close(server_fd);
        return;
    }

    std::cout << "[MJPEG] Server listening on port 8080 (/stream)\n";

    while (running) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(server_fd, &readfds);

        struct timeval tv;
        tv.tv_sec = 1;
        tv.tv_usec = 0;

        int activity = select(server_fd + 1, &readfds, NULL, NULL, &tv);
        if (activity < 0 || !running) break;
        if (activity == 0) continue; // timeout

        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_socket = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_socket < 0) continue;

        // 클라이언트 소켓은 blocking 모드로 복구
        int flags = fcntl(client_socket, F_GETFL, 0);
        fcntl(client_socket, F_SETFL, flags & ~O_NONBLOCK);

        std::cout << "[MJPEG] Client connected!\n";

        const char* http_header = "HTTP/1.1 200 OK\r\n"
                                  "Cache-Control: no-cache, private\r\n"
                                  "Pragma: no-cache\r\n"
                                  "Content-Type: multipart/x-mixed-replace; boundary=FRAME\r\n\r\n";
        send(client_socket, http_header, strlen(http_header), MSG_NOSIGNAL);

        while (running) {
            cv::Mat frame_copy;
            {
                std::lock_guard<std::mutex> lock(latest_frame_mutex);
                if (!latest_display_frame.empty()) {
                    frame_copy = latest_display_frame.clone();
                }
            }

            if (!frame_copy.empty()) {
                std::vector<uchar> buf;
                cv::imencode(".jpg", frame_copy, buf);

                char header[128];
                snprintf(header, sizeof(header), "--FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n", buf.size());
                
                if (send(client_socket, header, strlen(header), MSG_NOSIGNAL) < 0) break;
                if (send(client_socket, buf.data(), buf.size(), MSG_NOSIGNAL) < 0) break;
                if (send(client_socket, "\r\n", 2, MSG_NOSIGNAL) < 0) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50)); // ~20 FPS limit
        }
        std::cout << "[MJPEG] Client disconnected.\n";
        close(client_socket);
    }
    close(server_fd);
}

// ── 2. 웹소켓 클라이언트 (간이 구현) ──────────────────────────────
void websocket_client_func() {
    const char* server_ip = "192.168.1.10";
    int port = 8081;

    while (running) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            continue;
        }

        struct sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(port);
        if (inet_pton(AF_INET, server_ip, &serv_addr.sin_addr) <= 0) {
            close(sock);
            std::this_thread::sleep_for(std::chrono::seconds(3));
            continue;
        }

        std::cout << "[WS] Connecting to " << server_ip << ":" << port << "...\n";
        if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
            std::cout << "[WS] Connection failed. Retrying in 3s...\n";
            close(sock);
            
            // 종료 확인을 위해 1초씩 3번 쪼개서 Sleep
            for (int i=0; i<3 && running; ++i) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            continue;
        }

        // WebSocket Handshake 전송
        const char* handshake = 
            "GET /detection HTTP/1.1\r\n"
            "Host: 192.168.1.10:8081\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n";
        
        send(sock, handshake, strlen(handshake), MSG_NOSIGNAL);

        // Handshake 응답 무시 (간이 구현)
        char buf[1024];
        recv(sock, buf, sizeof(buf), 0);
        std::cout << "[WS] Connected successfully! Auto-reconnect active.\n";

        // Non-blocking 모드로 변경하여 루프에서 running 체크 가능하게 함
        fcntl(sock, F_SETFL, O_NONBLOCK);
        
        int alive_counter = 0;

        while (running) {
            unsigned char header[2];
            int ret = recv(sock, header, 2, 0);
            if (ret < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    alive_counter++;
                    if (alive_counter >= 100) { // 100 * 100ms = 10초마다 생존 로그 출력
                        std::cout << "[WS] Server connection is alive.\n";
                        alive_counter = 0;
                    }
                    continue;
                }
                break; // Error
            } else if (ret == 0) {
                break; // Closed
            }
            alive_counter = 0; // 데이터 수신 시 카운터 초기화

            int opcode = header[0] & 0x0F;
            int payload_len = header[1] & 0x7F;

            if (opcode == 0x8) { // Close frame
                break;
            }

            // 간소화: 확장 길이 무시하고 작은 JSON 메시지만 수신한다고 가정
            if (payload_len > 0 && payload_len < 126) {
                char* payload = new char[payload_len + 1];
                int received = 0;
                
                // blocking 모드로 잠시 전환하여 payload 끝까지 수신
                fcntl(sock, F_SETFL, fcntl(sock, F_GETFL, 0) & ~O_NONBLOCK);
                while (received < payload_len) {
                    int r = recv(sock, payload + received, payload_len - received, 0);
                    if (r <= 0) break;
                    received += r;
                }
                fcntl(sock, F_SETFL, O_NONBLOCK);

                payload[payload_len] = '\0';
                std::string msg(payload);
                delete[] payload;

                // 단순 문자열 검색으로 JSON 파싱 대체
                if (msg.find("\"ON\"") != std::string::npos) {
                    std::cout << "[WS] Received ON command -> Updating Hardware Status\n";
                    if (!camera_active_flags.empty()) {
                        camera_active_flags[0]->store(true);
                    }
                } else if (msg.find("\"OFF\"") != std::string::npos) {
                    std::cout << "[WS] Received OFF command -> Updating Hardware Status\n";
                    if (!camera_active_flags.empty()) {
                        camera_active_flags[0]->store(false);
                    }
                }
            }
        }

        std::cout << "[WS] Disconnected. Reconnecting...\n";
        close(sock);
    }
}
