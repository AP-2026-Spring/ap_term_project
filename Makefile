CC       := g++
CFLAGS   := -std=c++11 -I/usr/include -I/usr/local/include/opencv4
CXXFLAGS += -I/usr/include/opencv4

INCLUDEPATHS := -I${HOME}/project/tensorflow \
                -I${HOME}/project/EAI/yolo_with_pycam

LDPATH := -L${HOME}/project/tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib \
          -L${HOME}/project/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build

LDFLAGS := -pthread \
           -ltensorflow-lite \
           -lflatbuffers \
           -lopencv_core \
           -lopencv_highgui \
           -lopencv_imgproc \
           -lopencv_objdetect \
           -lopencv_imgcodecs \
           -lopencv_videoio \
           -lopencv_dnn \
           -l:libedgetpu.so.1.0 \
           -ldl

SRCS := yolo_with_pycam.cc \
        shared_state.cc \
        camera_thread.cc \
        inference_thread.cc \
        polling_thread.cc

OBJS := $(SRCS:.cc=.o)
EXEC := yolo_with_pycam

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDEPATHS) -o $@ $^ $(LDPATH) $(LDFLAGS)

%.o: %.cc
	$(CC) $(CFLAGS) $(CXXFLAGS) $(INCLUDEPATHS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)
