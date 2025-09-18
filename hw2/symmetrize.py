#!/usr/bin/env python3
"""
symmetrize.py -- intersect / grow-diag symmetrization for forward and reverse align files.

This module implements symmetrization methods to combine forward and reverse word alignments
into a single bidirectional alignment. IBM Model 1 is inherently asymmetric (models P(f|e)
but not P(e|f)), so we train two models and combine their predictions.

Assignment extension: "Train a French-English model and an English-French model and 
combine their predictions" - We train forward (model1_forward_1k.a) and reverse 
(model1_reverse_1k.a) models, then combine them using symmetrization methods.

Two symmetrization methods are implemented:
1. Intersection: Only keep alignments that appear in both forward and reverse models
2. Grow-diag: Start with intersection, then grow diagonally to add adjacent alignments

The grow-diag method typically performs better as it balances precision (intersection)
with recall (diagonal growth).
"""
import sys

def read_alignments_file(path):
    """
    Parse alignment file and return list of alignment pairs per sentence.
    
    Alignment files contain one line per sentence pair, with space-separated
    alignment pairs in "i-j" format, where i is foreign word index and j is English word index.
    
    Args:
        path (str): Path to alignment file
    
    Returns:
        list: List of lists, where each inner list contains (i,j) tuples representing
              alignments for one sentence pair
    """
    out = []
    
    # Read alignment file line by line
    with open(path, encoding="utf8") as fh:
        for line in fh:
            pairs = []  # Store alignments for this sentence
            s = line.strip()
            
            # Handle empty lines (empty sentences)
            if not s:
                out.append(pairs)
                continue
            
            # Parse each alignment pair in the line
            for token in s.split():
                # Skip malformed tokens (must contain "-")
                if "-" not in token:
                    continue
                
                # Split "i-j" into foreign index i and English index j
                a, b = token.split("-")
                pairs.append((int(a), int(b)))
            
            out.append(pairs)
    
    return out

def write_alignments(alist):
    """
    Output alignments in standard format.
    
    Converts list of alignment pairs back to the standard "i-j" format
    and prints one sentence per line.
    
    Args:
        alist (list): List of lists of (i,j) alignment tuples
    """
    for pairs in alist:
        # Convert (i,j) tuples to "i-j" strings and sort for consistent output
        print(" ".join(f"{i}-{j}" for i, j in sorted(pairs)))

def intersection_forward_reverse(forward, reverse):
    """
    Compute intersection of forward and reverse alignments.
    
    Only keeps alignments that appear in both the forward model (P(f|e)) and
    reverse model (P(e|f)). This ensures high precision but may have low recall
    since it's very conservative.
    
    Args:
        forward (list): Forward alignments from P(f|e) model
        reverse (list): Reverse alignments from P(e|f) model (coordinates swapped)
    
    Returns:
        list: Intersection alignments, one list per sentence
    """
    out = []
    
    # Process each sentence pair
    for f_pairs, r_pairs in zip(forward, reverse):
        # Convert reverse alignments back to forward coordinate system
        # Reverse model gives (j,i) pairs, we need (i,j) for consistency
        r_set = set((j, i) for i, j in r_pairs)
        f_set = set(f_pairs)
        
        # Intersection: only alignments that appear in both models
        out.append(sorted(f_set & r_set))
    
    return out

def grow_diag(forward, reverse):
    """
    Implement grow-diag symmetrization algorithm.
    
    This method combines the precision of intersection with the recall of diagonal growth:
    1. Start with intersection of forward and reverse alignments
    2. Grow diagonally by adding adjacent alignments from the union
    3. Continue until no more adjacent alignments can be added
    
    The diagonal growth helps recover missing alignments while maintaining connectivity
    and avoiding spurious long-distance alignments.
    
    Args:
        forward (list): Forward alignments from P(f|e) model
        reverse (list): Reverse alignments from P(e|f) model (coordinates swapped)
    
    Returns:
        list: Grow-diag symmetrized alignments, one list per sentence
    """
    out = []
    
    # Process each sentence pair
    for f_pairs, r_pairs in zip(forward, reverse):
        # Convert to sets for efficient operations
        f_set = set(f_pairs)
        r_set = set((j, i) for i, j in r_pairs)  # Convert reverse coordinates
        
        # Compute union and intersection
        union = f_set | r_set  # All possible alignments from either model
        inter = f_set & r_set  # High-confidence alignments (both models agree)
        
        # Handle empty case
        if not union:
            out.append([])
            continue
        
        # Start with intersection (high precision)
        added = set(inter)
        
        # Define 4-connected neighbors for diagonal growth
        # These represent adjacent positions in the alignment matrix
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        # Iteratively grow diagonally
        changed = True
        while changed:
            changed = False
            
            # For each current alignment, try to add adjacent alignments
            for (i, j) in list(added):  # Use list() to avoid modifying set during iteration
                for di, dj in neighbors:
                    ni, nj = i + di, j + dj  # Neighbor position
                    
                    # Skip if already added
                    if (ni, nj) in added:
                        continue
                    
                    # Add if it exists in the union (either model predicted it)
                    if (ni, nj) in union:
                        added.add((ni, nj))
                        changed = True  # Continue growing
        
        # Convert back to sorted list for output
        out.append(sorted(added))
    
    return out

def main():
    """
    Main function: parse arguments and run symmetrization.
    
    Command-line interface for combining forward and reverse alignments.
    Supports both intersection and grow-diag methods.
    """
    # Check for minimum required arguments
    if len(sys.argv) < 3:
        print("Usage: python symmetrize.py forward.align reverse.align [--method grow-diag|intersect]")
        sys.exit(1)
    
    # Parse symmetrization method (default: grow-diag)
    method = "grow-diag"
    if "--method" in sys.argv:
        idx = sys.argv.index("--method")
        if idx + 1 < len(sys.argv):
            method = sys.argv[idx + 1]
    
    # Load forward and reverse alignment files
    forward = read_alignments_file(sys.argv[1])
    reverse = read_alignments_file(sys.argv[2])
    
    # Apply selected symmetrization method
    if method == "intersect":
        out = intersection_forward_reverse(forward, reverse)
    else:  # grow-diag (default)
        out = grow_diag(forward, reverse)
    
    # Output symmetrized alignments
    write_alignments(out)

if __name__ == "__main__":
    main()