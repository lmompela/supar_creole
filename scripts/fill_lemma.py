#!/usr/bin/env python3
import sys

def fill_lemma(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.rstrip("\n")
            # If it's a comment line or blank line, write it as is.
            if not stripped or stripped.startswith("#"):
                fout.write(stripped + "\n")
                continue
            
            # Split the line into fields (CoNLL-U should have 10 columns)
            fields = stripped.split("\t")
            if len(fields) < 3:
                # Not a standard token line; write as is.
                fout.write(stripped + "\n")
                continue
            
            # If lemma (third column) is '_' or empty, fill it with FORM (second column)
            if fields[2].strip() == "_" or fields[2].strip() == "":
                fields[2] = fields[1]
            
            fout.write("\t".join(fields) + "\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python fill_lemma.py <input_conllu_file> <output_conllu_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    fill_lemma(input_file, output_file)
    print(f"Processed file saved to {output_file}")

if __name__ == "__main__":
    main()
