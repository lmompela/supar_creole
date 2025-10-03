#!/usr/bin/env python3
import sys

def read_file_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def compare_files(file1, file2):
    lines1 = read_file_lines(file1)
    lines2 = read_file_lines(file2)
    max_lines = max(len(lines1), len(lines2))
    differences = []
    for i in range(max_lines):
        # Get line i from each file, or a marker if the file is shorter.
        line1 = lines1[i] if i < len(lines1) else "<NO LINE>"
        line2 = lines2[i] if i < len(lines2) else "<NO LINE>"
        if line1 != line2:
            differences.append((i + 1, line1, line2))
    return differences

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_files.py <file1> <file2>")
        sys.exit(1)
    file1, file2 = sys.argv[1], sys.argv[2]
    diffs = compare_files(file1, file2)
    if not diffs:
        print("The files are identical.")
    else:
        print(f"Found {len(diffs)} differences:")
        for lineno, l1, l2 in diffs:
            print(f"Line {lineno}:")
            print(f"  {file1}: {l1}")
            print(f"  {file2}: {l2}")
            print("-" * 40)

if __name__ == "__main__":
    main()
