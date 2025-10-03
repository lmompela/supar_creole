#!/usr/bin/env python3
import sys

def check_embedding_file(embed_path, expected_dim=100, sep=" "):
    with open(embed_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            parts = line.strip().split(sep)
            # Skip lines that might be header (if any)
            if len(parts) <= 1:
                continue
            # The first part is the token, the rest are the vector dimensions.
            vec = parts[1:]
            if len(vec) != expected_dim:
                print(f"Line {i} error: Expected {expected_dim} dimensions, got {len(vec)}. Line: {line.strip()}")
                return
    print("All lines have the expected number of dimensions.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_embed.py <embedding_file>")
        sys.exit(1)
    check_embedding_file(sys.argv[1])
