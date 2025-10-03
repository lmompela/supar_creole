#!/usr/bin/env python3
import sys

def fix_file_eof(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as fin:
        content = fin.read()
    # Ensure the file ends with a newline
    if not content.endswith("\n"):
        content += "\n"
    # Now check if the last non-empty line is followed by an empty line.
    # Split lines while keeping newlines.
    lines = content.splitlines()
    # If the last line (after stripping) is not empty, then add an extra newline.
    if lines and lines[-1].strip():
        content += "\n"
    with open(output_file, "w", encoding="utf-8") as fout:
        fout.write(content)
    print(f"File '{input_file}' fixed and saved to '{output_file}'.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_conllu_eof.py <input_file> <output_file>")
        sys.exit(1)
    fix_file_eof(sys.argv[1], sys.argv[2])
