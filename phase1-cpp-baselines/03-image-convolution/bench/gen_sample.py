#!/usr/bin/env python3
"""Generate a synthetic 1024x1024 grayscale PGM test image (no external deps)."""
import struct

W, H = 1024, 1024
path = "bench/sample.pgm"

with open(path, "wb") as f:
    f.write(f"P5\n{W} {H}\n255\n".encode())
    row = bytearray(W)
    for y in range(H):
        for x in range(W):
            # gradient + checkerboard + a bit of structure, so blur is visible
            val = ((x * 255) // W + (y * 255) // H) // 2
            if (x // 32 + y // 32) % 2 == 0:
                val = 255 - val
            row[x] = val & 0xFF
        f.write(row)

print(f"wrote {path} ({W}x{H})")
