#!/usr/bin/env python3
from __future__ import annotations
import sys
from supar.models.dep.biaffine.transform import CoNLL, CoNLLSentence

def debug_conll_reader(file_path: str):
    # Create a CoNLL transform instance
    transform = CoNLL()
    
    print(f"Reading file: {file_path}")
    sentences = []
    sentence_token_counts = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        sentence = []
        index = 0
        for line in f:
            line = line.strip()
            if len(line) == 0:
                if sentence:  # Only yield if there is something accumulated
                    sent_obj = CoNLLSentence(transform, sentence, index)
                    token_count = len(sent_obj.values[0]) if sent_obj.values else 0
                    print(f"Sentence {index}: {token_count} tokens")
                    sentences.append(sent_obj)
                    sentence_token_counts.append(token_count)
                    index += 1
                    sentence = []
            else:
                sentence.append(line)
        # After reading all lines, if there's a leftover sentence, process it.
        if sentence:
            sent_obj = CoNLLSentence(transform, sentence, index)
            token_count = len(sent_obj.values[0]) if sent_obj.values else 0
            print(f"Sentence {index}: {token_count} tokens (final sentence)")
            sentences.append(sent_obj)
            sentence_token_counts.append(token_count)
            index += 1

    print(f"\nTotal number of sentences: {len(sentences)}\n")
    if sentences:
        print("First sentence in CoNLL-X format:")
        print(sentences[0].conll_format())
    
    return sentences

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test_conll_reader.py <conll_file>")
        sys.exit(1)
    file_path = sys.argv[1]
    debug_conll_reader(file_path)
