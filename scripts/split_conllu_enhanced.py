#!/usr/bin/env python3
import argparse
import random
import os

def read_conllu_file(filepath):
    """
    Reads the .conllu file and returns a list of sentences.
    Each sentence is defined as a block of text separated by a blank line.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    # Split on double newlines to get sentences
    sentences = content.split("\n\n")
    return sentences

def write_conllu_file(sentences, filepath):
    """
    Writes a list of sentences to a .conllu file.
    Sentences are separated by double newlines.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(sentences))
        f.write("\n")

def write_markdown_report(args, total_sentences, train_count, dev_count, test_count):
    """
    Writes a markdown report with argument and split summary.
    The markdown file is saved in the same directory as the training output file,
    with the same basename and a .md extension.
    """
    # Determine the markdown file path based on the training output file location.
    base, _ = os.path.splitext(args.train_out)
    md_filepath = base + ".md"

    md_content = f"""# Dataset Split Report

**Input File:** `{args.input_file}`  
**Ratios:** `{args.ratios}`  
**Random Seed:** `{args.seed}`  

## Sentence Counts

- **Total sentences:** {total_sentences}
- **Training set:** {train_count} sentences  
  *Output file:* `{args.train_out}`
- **Development set:** {dev_count} sentences  
  *Output file:* `{args.dev_out}`
- **Test set:** {test_count} sentences  
  *Output file:* `{args.test_out}`

"""

    with open(md_filepath, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Markdown report written to: {md_filepath}")

def main():
    parser = argparse.ArgumentParser(
        description="Split a .conllu file into train, dev, and test sets."
    )
    parser.add_argument("input_file", help="Path to the input .conllu file")
    parser.add_argument(
        "--ratios",
        default="8,1,1",
        help="Comma-separated split ratios for train, dev, test (e.g., '8,1,1' or '3,3,3'). Default is 8,1,1."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--train_out",
        default="train.conllu",
        help="Output file name for the training set (default: train.conllu)"
    )
    parser.add_argument(
        "--dev_out",
        default="dev.conllu",
        help="Output file name for the development set (default: dev.conllu)"
    )
    parser.add_argument(
        "--test_out",
        default="test.conllu",
        help="Output file name for the test set (default: test.conllu)"
    )
    args = parser.parse_args()

    # Parse ratios and compute the sum for scaling
    try:
        train_ratio, dev_ratio, test_ratio = map(int, args.ratios.split(','))
    except ValueError:
        raise ValueError("Please provide the ratios as three comma-separated integers, e.g., '8,1,1'.")
    total_ratio = train_ratio + dev_ratio + test_ratio

    # Read sentences from the input file
    sentences = read_conllu_file(args.input_file)
    if not sentences:
        raise ValueError("No sentences found in the input file.")

    # Shuffle the sentences to ensure random splitting
    random.seed(args.seed)
    random.shuffle(sentences)

    n = len(sentences)
    train_count = round(n * train_ratio / total_ratio)
    dev_count = round(n * dev_ratio / total_ratio)
    # Ensure all sentences are assigned (remaining go to test)
    test_count = n - train_count - dev_count
    
    if test_ratio == 0:
        test_count = 0
        dev_count = n - train_count
    train_set = sentences[:train_count]
    dev_set = sentences[train_count:train_count + dev_count]
    test_set = sentences[train_count + dev_count:]

    # Write the splits to their respective output files
    write_conllu_file(train_set, args.train_out)
    write_conllu_file(dev_set, args.dev_out)
    write_conllu_file(test_set, args.test_out)

    # Print summary information
    print(f"Total sentences: {n}")
    print(f"Train set: {len(train_set)} sentences -> {args.train_out}")
    print(f"Dev set: {len(dev_set)} sentences -> {args.dev_out}")
    print(f"Test set: {len(test_set)} sentences -> {args.test_out}")
    print(f"Random Seed: {args.seed}")

    # Write the markdown report alongside the training file.
    write_markdown_report(args, n, len(train_set), len(dev_set), len(test_set))

if __name__ == "__main__":
    main()
