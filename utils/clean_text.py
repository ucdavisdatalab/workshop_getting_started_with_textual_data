#!/usr/bin/env python

"""
helper script to quickly clean files
"""

from argparse import ArgumentParser
import os, re

def remove_extra_chars(doc):
    """
    remove anything that isn't a letter
    """
    doc = re.sub(r"[-—_\.]", " ", doc)
    doc = re.sub(r"[^\w\s]", "", doc)
    doc = re.sub(f"[0-9]", " ", doc)
    doc = re.sub(r"\s+", " ", doc)
    return doc.lower()

def stop_text(doc, stopwords):
    """
    stop out a string
    """
    doc = doc.split()
    doc = [token for token in doc if token not in stopwords and len(token) > 2]
    doc = ' '.join(doc)
    return doc

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in_dir')
    parser.add_argument('--out_dir')
    parser.add_argument('--stopwords_file')
    args = parser.parse_args()

    stopwords = []
    with open(args.stopwords_file, 'r') as f:
        for word in f:
            stopwords.extend(word.split())

    fnames = os.listdir(args.in_dir)
    fnames = [fname for fname in fnames if fname.startswith(".") == False]

    for fname in fnames:
        with open(args.in_dir + fname, 'r') as f:
            text = f.read()
            text = remove_extra_chars(text)
            text = stop_text(text, stopwords)
            with open(args.out_dir + fname, 'w') as o:
                o.write(text)
            print("Finished", fname)

