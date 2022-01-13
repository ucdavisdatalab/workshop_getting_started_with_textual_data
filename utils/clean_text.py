#! /usr/bin/env python

"""
helper script to quickly clean files
"""

import pandas as pd
import re

with open("../data/voyant_stoplist.txt", 'r') as f:
    stopwords = f.read().split()

def remove_extra_chars(doc):
    """
    remove anything that isn't a letter
    """
    doc = re.sub(r"[-—_\.]", " ", doc)
    doc = re.sub(r"[^\w\s]", "", doc)
    doc = re.sub(f"[0-9]", " ", doc) 
    doc = re.sub(r"\s+", " ", doc)
    return doc

def apply_stopwords(doc):
    """
    stop out a string
    """
    doc = doc.split()
    doc = [token for token in doc if token not in stopwords and len(token) > 2]
    return ' '.join(doc)

def clean(doc):
    """
    clean a string; returns full strings (not bags-of-words)
    """
    doc = remove_extra_chars(doc.lower())
    doc = apply_stopwords(doc)
    return doc

print("Enter an input directory path")
indir = input()

print("\nEnter an output directory path")
outdir = input()

print(f"\nLoading files from {indir} and outputting them to {outdir}\n")
manifest = pd.read_csv(indir + "manifest.csv", index_col=0)

for fname in manifest['FILE_NAME']:
    with open(indir + fname, 'r') as f:
        text = f.read()
        cleaned = clean(text)
        with open(outdir + fname, 'w') as o:
            o.write(cleaned)
        print("Finished", fname)
