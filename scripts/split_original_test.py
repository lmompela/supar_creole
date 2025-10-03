import random
from pathlib import Path
random.seed(42)
def read_conllu_sentences(conllu_path):
    with open(conllu_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Split sentences by double newline, preserving structure
    sentences = [s.strip() for s in content.strip().split('\n\n') if s.strip()]
    return sentences

def write_conllu_sentences(sentences, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n\n')

def split_conllu(conllu_path, test_pct, output_prefix):
    sentences = read_conllu_sentences(conllu_path)
    random.shuffle(sentences)

    test_size = int(len(sentences) * test_pct)
    test_sentences = sentences[:test_size]
    remaining_sentences = sentences[test_size:]

    output_dir = Path(conllu_path).parent
    write_conllu_sentences(test_sentences, output_dir / f"{output_prefix}_test.conllu")
    write_conllu_sentences(remaining_sentences, output_dir / f"{output_prefix}_train_dev.conllu")

    return len(test_sentences), len(remaining_sentences)

# Example usage:
#split_conllu("../martinican_creole_parser_gold.conllu", test_pct=0.25, output_prefix="mc_original")

# Uncomment and adjust the path to use
test_count, remaining_count = split_conllu("ht_autogramm-ud.conllu", 0.25, "hc_original_split")
print(f"Test set: {test_count} sentences\nRemaining (train+dev): {remaining_count} sentences")
