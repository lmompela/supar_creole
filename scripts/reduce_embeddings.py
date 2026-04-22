"""
Reduce fastText .vec embeddings from 300d to target_dim using PCA.
Usage: python3 scripts/reduce_embeddings.py embeddings/cc.ht.300.vec embeddings/cc.ht.100.vec --dim 100
"""
import argparse
import numpy as np
from sklearn.decomposition import PCA

def reduce_vec(input_path, output_path, target_dim):
    print(f"Loading {input_path} ...")
    words, vectors = [], []
    with open(input_path, encoding='utf-8') as f:
        header = f.readline().split()
        n_words, orig_dim = int(header[0]), int(header[1])
        for line in f:
            parts = line.rstrip().split(' ')
            words.append(parts[0])
            vectors.append(list(map(float, parts[1:])))
    
    print(f"  {n_words} words, {orig_dim}d → fitting PCA to {target_dim}d ...")
    X = np.array(vectors, dtype=np.float32)
    pca = PCA(n_components=target_dim, random_state=42)
    X_reduced = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  Explained variance retained: {explained:.1f}%")

    print(f"  Writing {output_path} ...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"{n_words} {target_dim}\n")
        for word, vec in zip(words, X_reduced):
            f.write(word + ' ' + ' '.join(f'{v:.6f}' for v in vec) + '\n')
    print("  Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--dim', type=int, default=100)
    args = parser.parse_args()
    reduce_vec(args.input, args.output, args.dim)
