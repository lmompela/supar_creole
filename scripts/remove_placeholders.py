#!/usr/bin/env python3
import sys

def is_placeholder_sentence(sentence_lines):
    """
    Returns True if the sentence appears to be a placeholder, i.e.:
      - Contains a comment line exactly "# text = ##########"
      - Contains a token line whose second field is exactly "##########"
    """
    found_comment = False
    found_token = False
    for line in sentence_lines:
        line = line.strip()
        if line.startswith("# text"):
            # Check if the comment line exactly matches
            if line == "# text = ##########":
                found_comment = True
        elif line and not line.startswith("#"):
            # This is a token line; CoNLL token lines are tab-delimited.
            fields = line.split("\t")
            if len(fields) >= 2 and fields[1].strip() == "##########":
                found_token = True
    return found_comment and found_token

def remove_placeholders(input_file, output_file):
    # Read entire file and split into sentences (by blank lines)
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
    # Assume sentences are separated by one or more blank lines.
    sentences = [s for s in content.split("\n\n") if s.strip()]
    
    filtered = []
    for sentence in sentences:
        lines = sentence.strip().splitlines()
        if not is_placeholder_sentence(lines):
            filtered.append(sentence)
        else:
            # Optionally, print or log which sentence was removed
            print("Removed a placeholder sentence:")
            print("\n".join(lines))
            print("-" * 40)
    
    # Write the cleaned sentences to the output file, preserving the blank lines
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(filtered) + "\n")
    print(f"Finished. Cleaned file saved as {output_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python remove_placeholders.py <input_file> <output_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    remove_placeholders(input_file, output_file)

if __name__ == "__main__":
    main()
