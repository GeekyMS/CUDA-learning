NVCC      := nvcc
SM        ?= 86
NVCCFLAGS := -arch=sm_$(SM) -O2 -lineinfo -std=c++17
TARGET    := reduction

SRC := src/main.cu

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

bench: $(TARGET)
	./$(TARGET) 67108864   # 64M elements, all 7 versions

profile: $(TARGET)
	ncu --set full --kernel-name "reduce6" ./$(TARGET) 67108864

clean:
	rm -f $(TARGET)

.PHONY: bench profile clean
