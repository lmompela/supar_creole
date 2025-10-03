#!/usr/bin/env python3
import sys
import difflib

def extract_tokens_list(conllu_file):
    """Extracts tokens (FORM column) from a CoNLL-U file into a list."""
    tokens = []
    with open(conllu_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 10:
                continue
            # Remove any spaces from the token (FORM, column 2)
            token = parts[1].replace(" ", "")
            tokens.append(token)
    return tokens

def main():
    if len(sys.argv) != 3:
        print("Usage: python diff_tokens.py <gold_file> <system_file>")
        sys.exit(1)
    gold_file = sys.argv[1]
    system_file = sys.argv[2]
    
    gold_tokens = extract_tokens_list(gold_file)
    system_tokens = extract_tokens_list(system_file)
    
    gold_concat = " ".join(gold_tokens)
    system_concat = " ".join(system_tokens)
    
    print("Gold tokens concatenation (first 100 characters):")
    print(gold_concat[:100])
    print("\nSystem tokens concatenation (first 100 characters):")
    print(system_concat[:100])
    
    if gold_tokens == system_tokens:
        print("\nThe token sequences match exactly.")
    else:
        print("\nDifferences between gold and system token sequences (word-level diff):")
        diff = difflib.unified_diff(
            gold_tokens, system_tokens,
            fromfile="gold", tofile="system",
            lineterm=""
        )
        for line in diff:
            print(line)

if __name__ == "__main__":
    main()

