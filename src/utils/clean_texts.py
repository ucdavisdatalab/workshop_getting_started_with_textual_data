#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

def convert_tag(tag: str) -> str:
    """Convert a TreeBank tag to a WordNet tag."""
    if tag.startswith('J'):
        tag = wordnet.ADJ
    elif tag.startswith('V'):
        tag = wordnet.VERB
    elif tag.startswith('N'):
        tag = wordnet.NOUN
    elif tag.startswith('R'):
        tag = wordnet.ADV
    else:
        tag = ''

    return tag

def lemmatize_word(word: str, tag: str) -> str:
    """Lemmatize a word."""
    if tag:
        return LEMMATIZER.lemmatize(word, pos = tag)

    return LEMMATIZER.lemmatize(word)

def lemmatize(doc: str) -> str:
    """Lemmatize an entire document."""
    tokenized = nltk.word_tokenize(doc)
    tagged = nltk.pos_tag(tokenized)
    tagged = [(word, convert_tag(tag)) for (word, tag) in tagged]
    lemmatized = [lemmatize_word(word, tag) for (word, tag) in tagged]

    return " ".join(lemmatized)

def regex_clean(doc: str) -> str:
    """Use regex patterns to remove punctuation, digits, and whitespace."""
    to_remove = {
        "hyphens": (r"[-]|[—]|[_]", " ")
        , "punct": (r"[^\w\s]", "")
        , "digit": (r"[0-9]", "")
        , "space": (r"\s+", " ")
    }
    for kind in to_remove:
        pattern, sub = to_remove[kind]
        doc = re.sub(pattern, sub, doc)
    
    return doc

def remove_stopwords(doc: str, stopwords: list[str]) -> str:
    """Remove stopwords from a document."""
    doc = doc.split()
    doc = [tok for tok in doc if tok not in stopwords]
    doc = [tok for tok in doc if len(tok) > 2]

    return " ".join(doc)

def clean(doc: str, stopwords: list[str]) -> str:
    """Clean a document."""
    doc = doc.lower()
    cleaned = regex_clean(doc)
    cleaned = remove_stopwords(cleaned, stopwords)

    return cleaned

def main(args: argparse.Namespace) -> None:
    """Run the script."""
    with args.stoplist.open('r') as fin:
        stopwords = fin.read().split("\n")

    for path in args.indir.glob("*.txt"):
        with path.open('r') as fin:
            doc = fin.read()

        lemmatized = lemmatize(doc)
        cleaned = clean(lemmatized, stopwords)

        outpath = args.outdir.joinpath(path.name)
        with outpath.open('w') as fout:
            fout.write(cleaned)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Clean a directory of texts"
    )
    parser.add_argument('--indir', type=Path, help="Input directory")
    parser.add_argument('--outdir', type=Path, help="Output directory")
    parser.add_argument('--stoplist', type=Path, help="Stop list file")
    args = parser.parse_args()

    LEMMATIZER = WordNetLemmatizer()
    main(args)
