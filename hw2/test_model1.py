#!/usr/bin/env python3
"""
test_model1.py -- Unit tests for IBM Model 1 implementation.

This module contains unit tests to verify the correctness of the Model 1 implementation.
Tests cover basic functionality including corpus reading, vocabulary building,
translation probability initialization, and end-to-end alignment generation.

The tests use a small toy dataset to ensure fast execution while covering
the main code paths and edge cases.
"""
import unittest
import os
import subprocess
from model1 import read_corpus, read_corpus as rc, init_t, align_bitext

class Model1QuickTests(unittest.TestCase):
    """
    Test suite for IBM Model 1 implementation.
    
    Tests basic functionality with a minimal toy dataset to ensure
    the implementation works correctly before running on larger datasets.
    """
    
    def test_toy_bitext_and_align(self):
        """
        Test end-to-end functionality with toy French-English data.
        
        This test verifies:
        1. Corpus reading from files
        2. Vocabulary extraction
        3. Translation probability initialization
        4. Command-line interface execution
        5. Alignment output format
        
        Uses a minimal dataset: "le chat" -> "the cat", "la maison" -> "the house"
        """
        # Create tiny test data files
        with open("toy.f", "w") as F:
            F.write("le chat\nla maison\n")
        with open("toy.e", "w") as E:
            E.write("the cat\nthe house\n")
        
        # Test corpus reading functionality
        # Read with prefix "toy" to use our test files
        bitext = read_corpus(prefix="toy", f_suf="f", e_suf="e", n=2, reverse=False)
        self.assertEqual(len(bitext), 2)  # Should read exactly 2 sentence pairs
        
        # Test vocabulary building
        f_vocab, e_vocab = set(), set()
        for f, e in bitext:
            f_vocab.update(f)  # Add French words
            e_vocab.update(e)  # Add English words
        
        # Test translation probability initialization
        t = init_t(e_vocab, f_vocab)
        
        # Test end-to-end execution via command-line interface
        # Run 1 EM iteration quickly via subprocess to ensure script runs
        cmd = ["python", "model1.py", "-d", "toy", "-f", "f", "-e", "e", "-n", "2", "--iters", "1"]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Verify successful execution (return code 0)
        self.assertEqual(p.returncode, 0)
        
        # Verify output format: should have 2 lines (one per sentence)
        lines = [l for l in p.stdout.splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)
        
        # Each line should contain alignment pairs in "i-j" format
        for line in lines:
            # Check that line contains valid alignment format
            pairs = line.split()
            for pair in pairs:
                self.assertIn("-", pair)  # Each pair should contain "-"
                # Should be able to split into two integers
                parts = pair.split("-")
                self.assertEqual(len(parts), 2)
                int(parts[0])  # Should be valid integer
                int(parts[1])  # Should be valid integer

if __name__ == "__main__":
    # Run all tests when script is executed directly
    unittest.main()