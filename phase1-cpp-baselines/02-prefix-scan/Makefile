CXX      := g++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -pthread
TARGET   := prefix_scan

SRC := src/main.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

bench: $(TARGET)
	@echo "=== N=1M ===" && ./$(TARGET) 1000000
	@echo "=== N=10M ===" && ./$(TARGET) 10000000

clean:
	rm -f $(TARGET)

.PHONY: bench clean
