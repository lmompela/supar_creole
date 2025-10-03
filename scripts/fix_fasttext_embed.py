#!/usr/bin/env python3
import sys

def fix_embedding_file(input_file: str, output_file: str, expected_dim: int = 100, sep: str = " "):
    fixed_lines = []
    with open(input_file, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(sep)
            token = parts[0]
            vec_parts = parts[1:]
            if len(vec_parts) == expected_dim:
                fixed_lines.append(line)
            elif len(vec_parts) < expected_dim:
                print(f"Line {i} error: Expected {expected_dim} dimensions, got {len(vec_parts)}. Padding with zeros.")
                # Pad missing dimensions with zeros
                vec_parts += ["0.0"] * (expected_dim - len(vec_parts))
                fixed_lines.append(token + sep + sep.join(vec_parts))
            else:
                print(f"Line {i} warning: Expected {expected_dim} dimensions, got {len(vec_parts)}. Truncating extra dimensions.")
                # Truncate if there are extra dimensions
                vec_parts = vec_parts[:expected_dim]
                fixed_lines.append(token + sep + sep.join(vec_parts))
    with open(output_file, "w", encoding="utf-8") as fout:
        for line in fixed_lines:
            fout.write(line + "\n")
    print(f"Fixed embeddings written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fix_embed.py <input_file> <output_file> [expected_dim]")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    expected_dim = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    fix_embedding_file(input_file, output_file, expected_dim)
