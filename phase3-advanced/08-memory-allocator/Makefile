NVCC      := nvcc
CXX       := g++
SM        ?= 86
NVCCFLAGS := -arch=sm_$(SM) -O2 -lineinfo -std=c++17
CXXFLAGS  := -std=c++17 -O2 -Wall -Wextra
TARGET    := allocator

SRC := src/main.cu

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

bench: $(TARGET)
	./$(TARGET) 10000000

clean:
	rm -f $(TARGET)

.PHONY: bench clean
