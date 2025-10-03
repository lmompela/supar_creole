#!/usr/bin/env python3
import sys

def extract_sentence_token_counts(conllu_file):
    counts = []
    with open(conllu_file, "r", encoding="utf-8") as f:
        sentence_lines = []
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                # Process the sentence if there are lines
                if sentence_lines:
                    token_count = sum(1 for l in sentence_lines if not l.startswith("#"))
                    counts.append(token_count)
                    sentence_lines = []
            else:
                sentence_lines.append(line)
        if sentence_lines:  # Process any remaining sentence
            token_count = sum(1 for l in sentence_lines if not l.startswith("#"))
            counts.append(token_count)
    return counts

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_sentence_counts.py <gold_file> <pred_file>")
        sys.exit(1)
    gold_file, pred_file = sys.argv[1], sys.argv[2]
    gold_counts = extract_sentence_token_counts(gold_file)
    pred_counts = extract_sentence_token_counts(pred_file)
    
    if len(gold_counts) != len(pred_counts):
        print(f"Number of sentences differ: Gold has {len(gold_counts)}, Predicted has {len(pred_counts)}")
    else:
        print(f"Both files have {len(gold_counts)} sentences.\n")
    
    for i, (g, p) in enumerate(zip(gold_counts, pred_counts)):
        if g != p:
            print(f"Sentence {i}: Gold has {g} tokens, Predicted has {p} tokens")
    print("\nDone comparing sentence token counts.")

if __name__ == "__main__":
    main()
