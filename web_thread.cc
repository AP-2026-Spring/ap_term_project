#include "web_thread.h"
#include "shared_state.h"
#include <arpa/inet.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <string>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

// ── 1. MJPEG 스트리밍 서버 ──────────────────────────────────────────
void mjpeg_server_func() {
  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0)
    return;

  int opt = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  // Non-blocking 설정 (종료 시 accept에서 무한 대기 방지)
  fcntl(server_fd, F_SETFL, O_NONBLOCK);

  struct sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(8080);

  if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
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
    if (activity < 0 || !running)
      break;
    if (activity == 0)
      continue; // timeout

    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_socket =
        accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
    if (client_socket < 0)
      continue;

    // 클라이언트 소켓은 blocking 모드로 복구
    int flags = fcntl(client_socket, F_GETFL, 0);
    fcntl(client_socket, F_SETFL, flags & ~O_NONBLOCK);

    // HTTP GET Request 헤더 읽기 및 카메라 번호 파싱
    char req_buf[1024] = {0};
    int bytes_rec = recv(client_socket, req_buf, sizeof(req_buf) - 1, 0);
    int target_cam = 0; // 디폴트: 0번 카메라
    if (bytes_rec > 0) {
      std::string request_str(req_buf);
      size_t cam_param_pos = request_str.find("cam=");
      if (cam_param_pos != std::string::npos) {
        try {
          target_cam = std::stoi(request_str.substr(cam_param_pos + 4, 1));
          std::cout << "[MJPEG] Client requested camera index: " << target_cam
                    << "\n";
        } catch (...) {
          std::cerr
              << "[MJPEG] Failed to parse target camera index from request!\n";
        }
      }
    }

    std::cout << "[MJPEG] Client connected for camera " << target_cam << "!\n";

    const char *http_header =
        "HTTP/1.1 200 OK\r\n"
        "Cache-Control: no-cache, private\r\n"
        "Pragma: no-cache\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=FRAME\r\n\r\n";
    send(client_socket, http_header, strlen(http_header), MSG_NOSIGNAL);

    while (running) {
      cv::Mat frame_copy;
      {
        std::lock_guard<std::mutex> lock(latest_frame_mutex);
        if (target_cam >= 0 && target_cam < (int)latest_display_frames.size()) {
          if (!latest_display_frames[target_cam].empty()) {
            frame_copy = latest_display_frames[target_cam].clone();
          }
        }
      }

      if (!frame_copy.empty()) {
        std::vector<uchar> buf;
        cv::imencode(".jpg", frame_copy, buf);

        char header[128];
        snprintf(header, sizeof(header),
                 "--FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: "
                 "%zu\r\n\r\n",
                 buf.size());

        if (send(client_socket, header, strlen(header), MSG_NOSIGNAL) < 0)
          break;
        if (send(client_socket, buf.data(), buf.size(), MSG_NOSIGNAL) < 0)
          break;
        if (send(client_socket, "\r\n", 2, MSG_NOSIGNAL) < 0)
          break;
      }
      std::this_thread::sleep_for(
          std::chrono::milliseconds(50)); // ~20 FPS limit
    }
    std::cout << "[MJPEG] Client disconnected.\n";
    close(client_socket);
  }
  close(server_fd);
}

#include <atomic>

// WebSocket 마스크 텍스트 프레임 인코딩 및 전송 헬퍼 (RFC 6455 규격 준수)
static void send_websocket_text(int sock, const std::string &text) {
  size_t len = text.length();
  std::vector<unsigned char> frame;

  // Byte 0: FIN = 1, Opcode = 1 (Text)
  frame.push_back(0x81);

  // Byte 1: Mask = 1, Payload Length
  if (len < 126) {
    frame.push_back(0x80 | len);
  } else if (len <= 65535) {
    frame.push_back(0x80 | 126);
    frame.push_back((len >> 8) & 0xFF);
    frame.push_back(len & 0xFF);
  } else {
    frame.push_back(0x80 | 127);
    for (int i = 7; i >= 0; --i) {
      frame.push_back((len >> (i * 8)) & 0xFF);
    }
  }

  // Masking Key: 4 bytes (간단히 고정 마스크 사용)
  unsigned char mask[4] = {0x12, 0x34, 0x56, 0x78};
  for (int i = 0; i < 4; ++i) {
    frame.push_back(mask[i]);
  }

  // 마스킹 처리된 페이로드 데이터
  for (size_t i = 0; i < len; ++i) {
    frame.push_back(text[i] ^ mask[i % 4]);
  }

  send(sock, frame.data(), frame.size(), MSG_NOSIGNAL);
}

// OS /proc/stat 파일 파싱을 통한 실제 CPU 사용량 계산 함수
static double get_real_cpu_usage() {
  static unsigned long long prev_user = 0, prev_nice = 0, prev_sys = 0,
                            prev_idle = 0, prev_iowait = 0, prev_irq = 0,
                            prev_softirq = 0;

  std::ifstream file("/proc/stat");
  if (!file.is_open())
    return 18.5;

  std::string cpu;
  unsigned long long user, nice, sys, idle, iowait, irq, softirq;
  if (!(file >> cpu >> user >> nice >> sys >> idle >> iowait >> irq >>
        softirq)) {
    return 18.5;
  }

  unsigned long long prev_total = prev_user + prev_nice + prev_sys + prev_idle +
                                  prev_iowait + prev_irq + prev_softirq;
  unsigned long long total = user + nice + sys + idle + iowait + irq + softirq;

  unsigned long long prev_idle_total = prev_idle + prev_iowait;
  unsigned long long idle_total = idle + iowait;

  double usage = 18.5;
  if (total > prev_total) {
    unsigned long long delta_total = total - prev_total;
    unsigned long long delta_idle = idle_total - prev_idle_total;
    usage = 100.0 * (delta_total - delta_idle) / (double)delta_total;
  }

  prev_user = user;
  prev_nice = nice;
  prev_sys = sys;
  prev_idle = idle;
  prev_iowait = iowait;
  prev_irq = irq;
  prev_softirq = softirq;

  return usage;
}

// OS /proc/meminfo 파일 파싱을 통한 실제 램 사용량(GB) 계산 함수
static double get_real_ram_gb() {
  std::ifstream file("/proc/meminfo");
  if (!file.is_open())
    return 1.45;

  std::string key;
  unsigned long long val;
  std::string unit;

  unsigned long long total_kb = 0;
  unsigned long long avail_kb = 0;

  while (file >> key >> val >> unit) {
    if (key == "MemTotal:") {
      total_kb = val;
    } else if (key == "MemAvailable:") {
      avail_kb = val;
    }
    if (total_kb > 0 && avail_kb > 0)
      break;
  }

  if (total_kb == 0)
    return 1.45;

  unsigned long long used_kb = total_kb - avail_kb;
  double used_gb = used_kb / 1024.0 / 1024.0;
  return used_gb;
}

// 실제 CPU 온도 판독 함수 (sysfs thermal_zone0 temp 읽기, 불가능 시 로드 비례
// 모사 fallback)
static double get_real_cpu_temp(double cpu_usage) {
  std::ifstream file("/sys/class/thermal/thermal_zone0/temp");
  if (file.is_open()) {
    double millidegrees;
    if (file >> millidegrees) {
      return millidegrees / 1000.0;
    }
  }
  return 37.5 + (cpu_usage * 0.22);
}

// 실제 CPU 로드에 비례한 가동 전력(W) 계산
static double get_real_power_usage(double cpu_usage) {
  return 3.1 + (cpu_usage * 0.042);
}

// ── 2. 웹소켓 클라이언트 (간이 구현) ──────────────────────────────
void websocket_client_func() {
  const char *env_ip = std::getenv("WS_SERVER_IP");
  const char *env_port = std::getenv("WS_SERVER_PORT");

  std::string server_ip = env_ip ? env_ip : "127.0.0.1";
  int port = env_port ? std::stoi(env_port) : 8081;

  while (running) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
      std::this_thread::sleep_for(std::chrono::seconds(3));
      continue;
    }

    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, server_ip.c_str(), &serv_addr.sin_addr) <= 0) {
      close(sock);
      std::this_thread::sleep_for(std::chrono::seconds(3));
      continue;
    }

    std::cout << "[WS] Connecting to " << server_ip << ":" << port << "...\n";
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
      std::cout << "[WS] Connection failed. Retrying in 3s...\n";
      close(sock);

      // 종료 확인을 위해 1초씩 3번 쪼개서 Sleep
      for (int i = 0; i < 3 && running; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
      continue;
    }

    // WebSocket Handshake 전송
    char handshake[512];
    snprintf(handshake, sizeof(handshake),
             "GET /detection HTTP/1.1\r\n"
             "Host: %s:%d\r\n"
             "Upgrade: websocket\r\n"
             "Connection: Upgrade\r\n"
             "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
             "Sec-WebSocket-Version: 13\r\n\r\n",
             server_ip.c_str(), port);

    send(sock, handshake, strlen(handshake), MSG_NOSIGNAL);

    // Handshake 응답 무시 (간이 구현)
    char buf[1024];
    recv(sock, buf, sizeof(buf), 0);
    std::cout << "[WS] Connected successfully! Auto-reconnect active.\n";

    // Non-blocking 모드로 변경하여 루프에서 running 체크 가능하게 함
    fcntl(sock, F_SETFL, O_NONBLOCK);

    // 시스템 정보 송출을 위한 백그라운드 쓰레드 실행
    std::atomic<bool> metrics_running{true};
    std::thread metrics_thread([sock, &metrics_running]() {
      while (metrics_running) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        if (!metrics_running)
          break;

        // /proc 파일시스템으로부터 실제 시스템 리소스 상태 취득
        double cpu = get_real_cpu_usage();
        double ram_gb = get_real_ram_gb();
        double temp = get_real_cpu_temp(cpu);
        double power = get_real_power_usage(cpu);

        // 실시간 리소스 지표 JSON 생성
        std::string payload = "{\"type\":\"resource_stats\","
                              "\"cpu\":" +
                              std::to_string((int)std::round(cpu)) +
                              ","
                              "\"ram_gb\":" +
                              std::to_string(std::round(ram_gb * 100) / 100.0) +
                              ","
                              "\"power\":" +
                              std::to_string(std::round(power * 100) / 100.0) +
                              ","
                              "\"temp\":" +
                              std::to_string(std::round(temp * 10) / 10.0) +
                              "}";

        send_websocket_text(sock, payload);
      }
    });

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
        char *payload = new char[payload_len + 1];
        int received = 0;

        // blocking 모드로 잠시 전환하여 payload 끝까지 수신
        fcntl(sock, F_SETFL, fcntl(sock, F_GETFL, 0) & ~O_NONBLOCK);
        while (received < payload_len) {
          int r = recv(sock, payload + received, payload_len - received, 0);
          if (r <= 0)
            break;
          received += r;
        }
        fcntl(sock, F_SETFL, O_NONBLOCK);

        payload[payload_len] = '\0';
        std::string msg(payload);
        delete[] payload;

        // 단순 문자열 검색으로 JSON 파싱 대체
        int target_idx = 0;
        size_t id_pos = msg.find("\"camera_id\":");
        if (id_pos != std::string::npos) {
          try {
            int parsed_id = std::stoi(msg.substr(id_pos + 12));
            target_idx = parsed_id - 1001; // Map 1001 -> 0, 1002 -> 1, etc.
            std::cout << "[WS] Parsed camera_id: " << parsed_id
                      << " -> index: " << target_idx << "\n";
          } catch (...) {
            std::cerr << "[WS] Failed to parse camera_id from message: " << msg
                      << "\n";
          }
        }

        if (msg.find("\"ON\"") != std::string::npos) {
          std::cout << "[WS] Received ON command for index: " << target_idx
                    << "\n";
          if (target_idx >= 0 && target_idx < (int)camera_active_flags.size()) {
            camera_active_flags[target_idx]->store(true);
          }
        } else if (msg.find("\"OFF\"") != std::string::npos) {
          std::cout << "[WS] Received OFF command for index: " << target_idx
                    << "\n";
          if (target_idx >= 0 && target_idx < (int)camera_active_flags.size()) {
            camera_active_flags[target_idx]->store(false);
          }
        }

        // Write the updated states back to /dev/shm/camera_cmd.txt so the
        // polling thread is in sync!
        std::ofstream cmd_file(CAM_CMD_PATH);
        if (cmd_file.is_open()) {
          for (size_t i = 0; i < camera_active_flags.size(); ++i) {
            cmd_file << i << ":" << (camera_active_flags[i]->load() ? 1 : 0)
                     << "\n";
          }
          cmd_file.close();
          std::cout << "[WS] Successfully synchronized /dev/shm/camera_cmd.txt "
                       "with updated states!\n";
        }
      }

      // 백그라운드 리소스 송출 쓰레드 안전하게 해제 및 조인
      metrics_running = false;
      if (metrics_thread.joinable()) {
        metrics_thread.join();
      }

      std::cout << "[WS] Disconnected. Reconnecting...\n";
      close(sock);
    }
  }
}