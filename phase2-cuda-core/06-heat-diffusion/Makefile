NVCC      := nvcc
SM        ?= 86
NVCCFLAGS := -arch=sm_$(SM) -O2 -lineinfo -std=c++17
TARGET    := heat

SRC := src/main.cu

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

bench: $(TARGET)
	@echo "=== 512x512 ===" && ./$(TARGET) 512 1000
	@echo "=== 1024x1024 ===" && ./$(TARGET) 1024 1000
	@echo "=== 2048x2048 ===" && ./$(TARGET) 2048 1000

animate: $(TARGET)
	./$(TARGET) 512 5000 --frames bench/frames/
	python3 bench/animate.py

profile: $(TARGET)
	ncu --set full ./$(TARGET) 2048 100

clean:
	rm -f $(TARGET)

.PHONY: bench animate profile clean
