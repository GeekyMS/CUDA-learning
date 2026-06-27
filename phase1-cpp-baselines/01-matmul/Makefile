CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra
TARGET   := matmul

SRC := src/main.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

bench: $(TARGET)
	@echo "=== N=256 ===" && ./$(TARGET) 256
	@echo "=== N=512 ===" && ./$(TARGET) 512
	@echo "=== N=1024 ===" && ./$(TARGET) 1024

clean:
	rm -f $(TARGET)

.PHONY: bench clean
