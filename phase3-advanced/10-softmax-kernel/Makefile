NVCC      := nvcc
SM        ?= 86
NVCCFLAGS := -arch=sm_$(SM) -O2 -lineinfo -std=c++17
TARGET    := softmax

SRC := src/main.cu

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

bench: $(TARGET)
	@for N in 1024 4096 8192 16384; do \
		echo "=== seq_len=$$N ===" && ./$(TARGET) $$N; \
	done

verify: $(TARGET)
	./$(TARGET) 4096 --dump bench/output.bin
	python3 bench/verify.py bench/output.bin

profile: $(TARGET)
	ncu --set full ./$(TARGET) 8192

clean:
	rm -f $(TARGET)

.PHONY: bench verify profile clean
