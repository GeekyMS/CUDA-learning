CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra
TARGET   := imgconv

SRC := src/main.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

bench: $(TARGET)
	@echo "=== 1024x1024, K=5 ===" && ./$(TARGET) bench/sample.pgm 5 1.0
	@echo "=== 1024x1024, K=15 ===" && ./$(TARGET) bench/sample.pgm 15 2.0

test: $(TARGET)
	./$(TARGET) bench/sample.pgm 3 1.0 --verify

clean:
	rm -f $(TARGET)

.PHONY: bench test clean
