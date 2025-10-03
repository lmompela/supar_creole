#!/usr/bin/env python3
import re

def validate_conllu_file(filepath):
    issues = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check newline consistency by splitting into lines
    lines = content.splitlines()
    for i, line in enumerate(lines, start=1):
        # Check if a line that should be blank really is blank (no spaces/tabs)
        if line.strip() == "" and line != "":
            issues.append(f"Line {i}: Blank line contains invisible characters.")
    
    # Split into sentence blocks using double newlines (allowing for any whitespace-only block)
    sentences = [block for block in re.split(r'\n\s*\n', content.strip()) if block.strip()]
    print("Number of sentence blocks:", len(sentences))
    
    # Validate each sentence block
    for idx, sent in enumerate(sentences, start=1):
        # Each sentence should have at least one token (ignoring comment lines starting with '#')
        token_lines = [line for line in sent.splitlines() if not line.strip().startswith("#")]
        if not token_lines:
            issues.append(f"Sentence {idx}: No token lines found.")
            continue
        for j, token_line in enumerate(token_lines, start=1):
            # In CoNLL-U, token lines should be split into 10 fields by tabs.
            fields = token_line.split("\t")
            if len(fields) < 10:
                issues.append(f"Sentence {idx}, token line {j}: Expected at least 10 columns, got {len(fields)}. Line content: {token_line}")
            # Optional: Check for extra trailing spaces in the token line.
            if token_line != token_line.rstrip():
                issues.append(f"Sentence {idx}, token line {j}: Trailing whitespace detected.")
    
    return sentences, issues

def main():
    filepath = "../data/mc/seed_4/mc_ud-333-seed4-dev.conllu"  # update as needed
    sentences, issues = validate_conllu_file(filepath)
    
    if issues:
        print("Validation issues found:")
        for issue in issues:
            print(" -", issue)
    else:
        print("No formatting issues found in the file.")
    
    print("Total sentences (validated):", len(sentences))

if __name__ == "__main__":
    main()
