# Edge AI 유해동물 탐지 시스템 - Edge Device (yolo_with_pycam)

## 프로젝트 개요
라즈베리파이 카메라(Pycam)를 활용하여 영상을 캡처하고, Edge TPU 상에서 TensorFlow Lite 기반의 YOLO 모델을 사용하여 유해동물을 실시간으로 탐지(인퍼런스)하고 결과를 파싱 및 시각화하는 엣지 단말용 프로그램입니다.

## 필요한 패키지
이 프로젝트를 빌드하고 실행하기 위해 다음 환경이 구성되어 있어야 합니다:
* **OpenCV**: 디렉터리 바로 아래 경로에 OpenCV가 있어야 합니다.
* **TensorFlow**: TensorFlow(TFLite)가 설치되어 있어야 합니다.
* (참고) **raspicam**: 라즈베리파이 카메라 사용을 위한 C++ API (`https://github.com/cedricve/raspicam.git`)

## 빌드 방법
터미널에서 현재 디렉터리(`ap_term_project`)로 이동한 후, `make` 명령어를 실행하여 소스 코드를 빌드합니다.

```bash
make clean
make all
```

## 실행 방법
빌드가 완료된 후, 제공된 쉘 스크립트를 사용하여 프로그램을 실행할 수 있습니다. 
(필요한 경우 `.env.sample`을 참고하여 `.env` 파일을 생성하면, 스크립트 실행 시 환경 변수로 자동 로드됩니다.)

```bash
# 실행 권한 부여 (최초 1회)
chmod +x run_tpu.sh

# 프로그램 실행
./run_tpu.sh
```

> 내부적으로 `./yolo_with_pycam best_full_integer_quant_edgetpu.tflite 1` 명령이 실행되며 추론을 시작합니다.
