# check_cycles.py
import sys
from io import StringIO
from conll17_ud_eval import load_conllu, UDError  # adjust the import path as needed

def check_cycles(file_path: str):
    problematic = []
    with open(file_path, "r", encoding="utf-8") as f:
        sentence_lines = []
        sentence_index = 0
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if sentence_lines:
                    text = "\n".join(sentence_lines) + "\n"
                    try:
                        _ = load_conllu(StringIO(text))
                    except UDError as e:
                        print(f"Cycle detected in sentence {sentence_index}: {e}")
                        print("\n".join(sentence_lines))
                        print("-" * 40)
                        problematic.append(sentence_index)
                    sentence_index += 1
                    sentence_lines = []
            else:
                sentence_lines.append(line)
        if sentence_lines:
            text = "\n".join(sentence_lines) + "\n"
            try:
                _ = load_conllu(StringIO(text))
            except UDError as e:
                print(f"Cycle detected in sentence {sentence_index}: {e}")
                print("\n".join(sentence_lines))
                print("-" * 40)
                problematic.append(sentence_index)
    print(f"Total problematic sentences: {len(problematic)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_cycles.py <predicted_conllu_file>")
        sys.exit(1)
    check_cycles(sys.argv[1])
