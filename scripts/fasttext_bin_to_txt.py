#!/usr/bin/env python3
import fasttext
import sys

def convert_fasttext_bin_to_txt(model_path, output_path, vocab_path=None):
    # Load the FastText binary model
    model = fasttext.load_model(model_path)
    
    # If a vocabulary file is provided, use it (one word per line);
    # otherwise, use the model’s built-in vocabulary.
    if vocab_path:
        with open(vocab_path, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
    else:
        # get_words() returns all words in the model's dictionary.
        words = model.get_words()
    
    with open(output_path, "w", encoding="utf-8") as fout:
        for word in words:
            vector = model.get_word_vector(word)
            vector_str = " ".join(map(str, vector))
            fout.write(f"{word} {vector_str}\n")
    print(f"Converted model saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fasttext_bin_to_txt.py <model.bin> <output.txt> [vocab.txt]")
        sys.exit(1)
    model_path = sys.argv[1]
    output_path = sys.argv[2]
    vocab_path = sys.argv[3] if len(sys.argv) > 3 else None
    convert_fasttext_bin_to_txt(model_path, output_path, vocab_path)
