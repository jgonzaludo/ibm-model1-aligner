#!/usr/bin/env python3
"""
model1.py -- IBM Model 1 aligner (sparse), with optional diagonal bias and reverse training.

IBM Model 1 is a probabilistic word alignment model that learns translation probabilities
P(f|e) between foreign words f and English words e using Expectation-Maximization (EM).
The model assumes each foreign word is generated independently from some English word
in the same sentence, with uniform alignment probabilities.

This implementation includes several extensions:
- Diagonal bias: Prefers alignments near the diagonal using position-based weights
  (Assignment extension: "Implement a model that prefers to align words close to the diagonal")
- Reverse training: Train P(e|f) instead of P(f|e) for symmetrization
- Pruning: Remove low-probability translation pairs to reduce memory usage
- Sparse representation: Only store non-zero translation probabilities

Usage examples:
  python model1.py -n 1000 --iters 5 > model1_1k.a
  python model1.py -n 1000 --iters 5 --reverse > model1_rev_1k.a
  python model1.py -n 1000 --iters 5 --lambda 2.0 --prune 1e-8 > model1_bias_1k.a
"""
import argparse
import math
import sys
from collections import defaultdict

# Small epsilon to prevent division by zero
EPS = 1e-12

def read_corpus(prefix="data/hansards", f_suf="f", e_suf="e", n=sys.maxsize, reverse=False):
    """
    Read parallel corpus from files and return as list of sentence pairs.
    
    Args:
        prefix (str): Base filename prefix (e.g., "data/hansards")
        f_suf (str): Suffix for foreign language file (e.g., "f" for French)
        e_suf (str): Suffix for English language file (e.g., "e" for English)
        n (int): Maximum number of sentences to read
        reverse (bool): If True, swap foreign and English (for reverse training)
    
    Returns:
        list: List of (foreign_sentence, english_sentence) tuples, where each
              sentence is a list of words
    """
    # Construct file paths
    fpath = f"{prefix}.{f_suf}"
    epath = f"{prefix}.{e_suf}"
    bitext = []
    
    # Read both files simultaneously, line by line
    with open(fpath, encoding="utf8") as ff, open(epath, encoding="utf8") as ee:
        for i, (fl, el) in enumerate(zip(ff, ee)):
            # Stop if we've read enough sentences
            if i >= n:
                break
            
            # Split each line into words (whitespace-separated)
            f_sent = fl.strip().split()
            e_sent = el.strip().split()
            
            # Optionally swap order for reverse training (P(e|f) instead of P(f|e))
            if reverse:
                bitext.append((e_sent, f_sent))
            else:
                bitext.append((f_sent, e_sent))
    
    return bitext

def build_vocabs(bitext):
    """
    Extract vocabulary sets from parallel corpus.
    
    Args:
        bitext (list): List of (foreign_sentence, english_sentence) tuples
    
    Returns:
        tuple: (foreign_vocab, english_vocab) as sets of unique words
    """
    f_vocab = set()
    e_vocab = set()
    
    # Collect all unique words from both languages
    for f_sent, e_sent in bitext:
        f_vocab.update(f_sent)  # Add all foreign words in this sentence
        e_vocab.update(e_sent)  # Add all English words in this sentence
    
    return f_vocab, e_vocab

def init_t(e_vocab, f_vocab):
    """
    Initialize translation probabilities uniformly.
    
    In IBM Model 1, we need to initialize P(f|e) for all word pairs.
    We start with uniform probabilities: P(f|e) = 1/|F| where |F| is foreign vocab size.
    
    Args:
        e_vocab (set): Set of English words
        f_vocab (set): Set of foreign words
    
    Returns:
        dict: Nested dict t[e][f] = P(f|e) with uniform initialization
    """
    # Use defaultdict to automatically create nested dictionaries
    t = defaultdict(lambda: defaultdict(float))
    
    # Handle edge case of empty vocabulary
    if len(f_vocab) == 0:
        return t
    
    # Uniform probability: each foreign word has equal chance of being aligned
    # to any English word
    uniform = 1.0 / max(len(f_vocab), 1)
    
    # Initialize all possible word pairs with uniform probability
    for e in e_vocab:
        for f in f_vocab:
            t[e][f] = uniform
    
    return t

def bias_weight(i, j, len_f, len_e, lambda_val):
    """
    Compute diagonal bias weight for word alignment.
    
    The diagonal bias encourages alignments near the diagonal of the alignment matrix,
    based on the observation that word order tends to be preserved across languages.
    
    Formula: exp(-λ * |pos_f - pos_e|)
    where pos_f = i/(len_f-1) and pos_e = j/(len_e-1) are normalized positions.
    
    Args:
        i (int): Position of foreign word in sentence (0-indexed)
        j (int): Position of English word in sentence (0-indexed)
        len_f (int): Length of foreign sentence
        len_e (int): Length of English sentence
        lambda_val (float): Bias strength parameter (0 = no bias, higher = stronger)
    
    Returns:
        float: Bias weight between 0 and 1 (1 = no bias)
    """
    # If no bias is requested, return neutral weight
    if not lambda_val or lambda_val <= 0.0:
        return 1.0
    
    # Normalize positions to [0,1] range
    # Handle single-word sentences by setting position to 0
    pos_f = i / max(1.0, len_f - 1) if len_f > 1 else 0.0
    pos_e = j / max(1.0, len_e - 1) if len_e > 1 else 0.0
    
    # Compute distance from diagonal (0 = perfect diagonal alignment)
    diff = abs(pos_f - pos_e)
    
    # Apply exponential decay: closer to diagonal = higher weight
    return math.exp(-lambda_val * diff)

def em_iteration(bitext, t, lambda_val=0.0, prune_threshold=None):
    """
    Perform one iteration of Expectation-Maximization algorithm.
    
    E-step: Compute expected counts of word alignments using current probabilities
    M-step: Update translation probabilities using expected counts
    
    Args:
        bitext (list): Parallel corpus as list of sentence pairs
        t (dict): Current translation probabilities t[e][f] = P(f|e)
        lambda_val (float): Diagonal bias parameter
        prune_threshold (float): Remove probabilities below this threshold (None = no pruning)
    
    Returns:
        dict: Updated translation probabilities
    """
    # E-STEP: Collect expected counts
    count = defaultdict(lambda: defaultdict(float))  # count[e][f] = expected count
    total = defaultdict(float)  # total[e] = sum of all counts for English word e
    
    # Process each sentence pair
    for f_sent, e_sent in bitext:
        len_f = len(f_sent)
        len_e = len(e_sent)
        
        # For each foreign word, compute alignment probabilities to all English words
        for i, f_word in enumerate(f_sent):
            denom = 0.0  # Normalization constant
            scores = []  # Store (english_word, score) pairs
            
            # Compute unnormalized scores for all possible English alignments
            for j, e_word in enumerate(e_sent):
                # Score = translation prob * diagonal bias
                score = t[e_word].get(f_word, 0.0) * bias_weight(i, j, len_f, len_e, lambda_val)
                scores.append((e_word, score))
                denom += score
            
            # Handle numerical instability: if all scores are near zero
            if denom <= EPS:
                # Use uniform distribution as fallback
                uniform = 1.0 / max(1, len_e)
                for e_word, _ in scores:
                    count[e_word][f_word] += uniform
                    total[e_word] += uniform
            else:
                # Normalize scores to get posterior probabilities
                for e_word, score in scores:
                    post = score / denom  # P(align f_word to e_word | sentence pair)
                    count[e_word][f_word] += post
                    total[e_word] += post
    
    # M-STEP: Update translation probabilities using expected counts
    for e_word, f_counts in count.items():
        # Normalize counts to get probabilities: P(f|e) = count(e,f) / total(e)
        denom = total[e_word] + EPS  # Add epsilon to prevent division by zero
        for f_word, val in f_counts.items():
            t[e_word][f_word] = val / denom
        
        # Optional pruning: remove low-probability translations to save memory
        if prune_threshold:
            to_delete = [f for f, p in t[e_word].items() if p < prune_threshold]
            for f in to_delete:
                del t[e_word][f]
    
    return t

def align_bitext(bitext, t, lambda_val=0.0):
    """
    Generate word alignments using Viterbi approximation.
    
    For each foreign word, find the English word with highest alignment score.
    This is an approximation to true Viterbi decoding since we don't consider
    the full alignment path, just individual word alignments.
    
    Args:
        bitext (list): Parallel corpus as list of sentence pairs
        t (dict): Translation probabilities t[e][f] = P(f|e)
        lambda_val (float): Diagonal bias parameter
    
    Returns:
        list: List of alignment strings, one per sentence (format: "0-1 1-0 2-2")
    """
    out_lines = []
    
    # Process each sentence pair
    for f_sent, e_sent in bitext:
        len_f = len(f_sent)
        len_e = len(e_sent)
        pairs = []  # Store alignment pairs for this sentence
        
        # For each foreign word, find best English alignment
        for i, f_word in enumerate(f_sent):
            best_j = 0  # Index of best English word
            best_score = -1.0  # Best alignment score found so far
            
            # Try aligning to each English word
            for j, e_word in enumerate(e_sent):
                # Score = translation probability * diagonal bias
                score = t[e_word].get(f_word, 0.0) * bias_weight(i, j, len_f, len_e, lambda_val)
                
                # Keep track of best alignment
                if score > best_score:
                    best_score = score
                    best_j = j
            
            # Record this alignment as "foreign_index-english_index"
            pairs.append(f"{i}-{best_j}")
        
        # Join all alignments for this sentence with spaces
        out_lines.append(" ".join(pairs))
    
    return out_lines

def main():
    """
    Main function: parse arguments, train model, and output alignments.
    
    Command-line interface for IBM Model 1 training and alignment generation.
    Supports various options for diagonal bias, reverse training, and pruning.
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="data/hansards", help="data prefix")
    parser.add_argument("-f", "--french", default="f", help="suffix for foreign language file")
    parser.add_argument("-e", "--english", default="e", help="suffix for English language file")
    parser.add_argument("-n", "--num_sentences", type=int, default=1000, help="number of sentences to process")
    parser.add_argument("--iters", type=int, default=5, help="number of EM iterations")
    parser.add_argument("--reverse", action="store_true", help="train reverse model P(e|f) instead of P(f|e)")
    parser.add_argument("--lambda", dest="lambda_val", type=float, default=0.0, 
                       help="diagonal bias lambda (0 disables, higher = stronger bias)")
    parser.add_argument("--prune", type=float, default=0.0, 
                       help="prune threshold for tiny translation probabilities")
    parser.add_argument("--show-top", type=str, default=None, 
                       help="print top translations for this English word to stderr")
    opts = parser.parse_args()

    # Load parallel corpus
    bitext = read_corpus(prefix=opts.data, f_suf=opts.french, e_suf=opts.english,
                         n=opts.num_sentences, reverse=opts.reverse)
    if not bitext:
        sys.stderr.write("No sentences read; check paths.\n")
        sys.exit(1)

    # Extract vocabularies and initialize translation probabilities
    f_vocab, e_vocab = build_vocabs(bitext)
    t = init_t(e_vocab, f_vocab)

    # Run EM training for specified number of iterations
    for it in range(opts.iters):
        sys.stderr.write(f"EM iter {it+1}/{opts.iters} on {len(bitext)} sentences\n")
        t = em_iteration(bitext, t, lambda_val=opts.lambda_val, prune_threshold=(opts.prune or None))

    # Optional: show top translations for debugging
    if opts.show_top:
        tops = sorted(t[opts.show_top].items(), key=lambda kv: -kv[1])[:20] if opts.show_top in t else []
        sys.stderr.write(f"Top translations for '{opts.show_top}': {tops}\n")

    # Generate and output alignments
    for line in align_bitext(bitext, t, lambda_val=opts.lambda_val):
        print(line)

if __name__ == "__main__":
    main()