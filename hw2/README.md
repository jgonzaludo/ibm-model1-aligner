There are three python programs here (`-h` for usage):

- `./align` aligns words.

- `./check-alignments` checks that the entire dataset is aligned, and
  that there are no out-of-bounds alignment points.

- `./score-alignments` computes alignment error rate.

The commands work in a pipeline. For instance:

   > ./align -t 0.9 -n 1000 | ./check | ./grade -n 5

The `data` directory contains a fragment of the Canadian Hansards,
aligned by Ulrich Germann:

- `hansards.e` is the English side.

- `hansards.f` is the French side.

- `hansards.a` is the alignment of the first 37 sentences. The 
  notation i-j means the word as position i of the French is 
  aligned to the word at position j of the English. Notation 
  i?j means they are probably aligned. Positions are 0-indexed.

---

# IBM Model 1 Implementation Usage

This implementation consists of two main components: the IBM Model 1 trainer (`model1.py`) and the symmetrization tool (`symmetrize.py`).

## Basic Usage

### Training IBM Model 1

Train a forward model (P(f|e)) on 1000 sentences with 5 EM iterations:
```bash
python model1.py -n 1000 --iters 5 > forward_alignments.a
```

Train a reverse model (P(e|f)) for symmetrization:
```bash
python model1.py -n 1000 --iters 5 --reverse > reverse_alignments.a
```

### Adding Diagonal Bias

Train with diagonal bias (λ=2.0) to prefer alignments near the diagonal:
```bash
python model1.py -n 1000 --iters 5 --lambda 2.0 > biased_alignments.a
```

### Memory Optimization

Use pruning to remove low-probability translations and reduce memory usage:
```bash
python model1.py -n 1000 --iters 5 --prune 1e-8 > pruned_alignments.a
```

### Custom Data Files

Specify custom data files (default expects `data/hansards.f` and `data/hansards.e`):
```bash
python model1.py -d data/europarl -f fr -e en -n 5000 --iters 10
```

## Symmetrization

### Grow-Diag Symmetrization (Recommended)

Combine forward and reverse alignments using the grow-diag algorithm:
```bash
python symmetrize.py forward_alignments.a reverse_alignments.a > symmetrized.a
```

### Intersection-Only Symmetrization

Use intersection method for high-precision, low-recall alignments:
```bash
python symmetrize.py forward_alignments.a reverse_alignments.a --method intersect > intersection.a
```

## Complete Pipeline Example

Train both directions and symmetrize:
```bash
# Train forward model
python model1.py -n 1000 --iters 5 --lambda 1.5 > forward.a

# Train reverse model  
python model1.py -n 1000 --iters 5 --lambda 1.5 --reverse > reverse.a

# Symmetrize using grow-diag
python symmetrize.py forward.a reverse.a > final_alignments.a
```

## Command-Line Options

You can run **model1.py** or **symmetrize.py** with the **--help** flag to see a complete list of command line options.

### Further Reading
Please refer to **Writeup.md** for an overview of our implementation, and **algorithm.pdf** for a mathematical description of our algorithm!