NVCC     := nvcc
SM       ?= 86
NVCCFLAGS := -arch=sm_$(SM) -O2 -lineinfo -std=c++17
TARGET   := saxpy

SRC := src/main.cu

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

bench: $(TARGET)
	@echo "=== N=10M ===" && ./$(TARGET) 10000000
	@echo "=== N=100M ===" && ./$(TARGET) 100000000

profile: $(TARGET)
	ncu --set full ./$(TARGET) 100000000

clean:
	rm -f $(TARGET)

.PHONY: bench profile clean
