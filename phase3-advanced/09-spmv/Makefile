NVCC      := nvcc
SM        ?= 86
NVCCFLAGS := -arch=sm_$(SM) -O2 -lineinfo -std=c++17
TARGET    := spmv

SRC := src/main.cu

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

bench: $(TARGET)
	./$(TARGET) bench/matrix.mtx

profile: $(TARGET)
	ncu --set full ./$(TARGET) bench/matrix.mtx

clean:
	rm -f $(TARGET)

.PHONY: bench profile clean
