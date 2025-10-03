import sys
import os
import re

def concatenate_conllu(output_file, *input_files):
    """
    Concatenates multiple CoNLL-U files into one, ensuring correct formatting,
    sequential `sent_id`, and preserving legacy sent_id with source markers.
    """
    sentence_counter = 1  # New sent_id starts from 1
    output_sentences = []

    for input_file in input_files:
        source_marker = os.path.basename(input_file)  # Use filename as marker
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            sentences = re.split(r'\n\s*\n', content)  # Split sentences correctly

            for sent in sentences:
                lines = sent.split('\n')
                new_lines = []
                legacy_sent_id = None
                
                for line in lines:
                    if line.startswith('# sent_id ='):
                        legacy_sent_id = line.split('=')[-1].strip()
                        new_lines.append(f'# legacy_sent_id = {legacy_sent_id}')
                        new_lines.append(f'# source = {source_marker}')
                        new_lines.append(f'# sent_id = {sentence_counter}')
                    elif line.startswith('# text =') or line.startswith('# text_fr ='):
                        new_lines.append(line)  # Keep original text lines
                    else:
                        new_lines.append(line)  # Keep annotation lines
                
                if legacy_sent_id is None:
                    new_lines.insert(0, f'# source = {source_marker}')
                    new_lines.insert(1, f'# sent_id = {sentence_counter}')
                
                output_sentences.append('\n'.join(new_lines))
                sentence_counter += 1  # Increment sent_id for the next sentence

    # Ensure proper formatting with blank lines between sentences
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write('\n\n'.join(output_sentences) + '\n\n')
    
    print(f'Concatenation complete! Output saved to {output_file}')

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python concatenate_conllu.py output_file.conllu input1.conllu input2.conllu [...]")
        sys.exit(1)
    
    output_filepath = sys.argv[1]
    input_filepaths = sys.argv[2:]
    
    concatenate_conllu(output_filepath, *input_filepaths)
