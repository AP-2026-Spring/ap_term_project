CC := g++
CFLAGS := -std=c++11 -I/usr/include -I/usr/local/include/opencv4
INCLUDEPATHS := -I${HOME}/project/tensorflow -I${HOME}/project/EAI/yolo_with_pycam
LDFLAGS := -pthread -ltensorflow-lite -lflatbuffers -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_imgcodecs -lopencv_videoio -l:libedgetpu.so.1.0 -ldl

LDPATH := -L${HOME}/project/tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib\
	  -L${HOME}/project/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build

CXXFLAGS += -I/usr/include/opencv4

SRCS := yolo_with_pycam.cc
OBJS := $(SRCS:.cc=.o)
EXEC := yolo_with_pycam

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDEPATHS) -o $@ $^ $(LDPATH) $(LDFLAGS)  

%.o: %.cc
	$(CC) $(CFLAGS) $(INCLUDEPATHS) -c $< -o $@   

clean : 
	rm -f $(OBJS) $(EXEC)
