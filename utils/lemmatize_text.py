#!/usr/bin/env python

"""
lemmatize a directory of texts. note: for the purposes of teaching 
this uses the treebank detokenizer to return texts to untokenized 
representations
"""

from argparse import ArgumentParser
import os
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

tokenizer = TreebankWordTokenizer()
detokenizer = TreebankWordDetokenizer()
lemmatizer = WordNetLemmatizer()

def convert_tag(tag):
    """
    convert treebank tags to wordnet tags
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def lemmatize_word(word, tag):
    """
    lemmatize using wordnet tags (when available)
    """
    if tag != '':
        return lemmatizer.lemmatize(word, pos=tag)
    else:
        return lemmatizer.lemmatize(word)

def lemmatize_text(doc):
    """
    tokenize a text, tag it, and then use those tags to lemmatize 
    the text
    """
    bow = tokenizer.tokenize(doc)
    tagged = nltk.pos_tag(bow)
    tagged = [(i[0], convert_tag(i[1])) for i in tagged]
    lemmatized = [lemmatize_word(i[0], i[1]) for i in tagged]
    return detokenizer.detokenize(lemmatized)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in_dir')
    parser.add_argument('--out_dir')
    args = parser.parse_args()

    fnames = os.listdir(args.in_dir)
    fnames = [fname for fname in fnames if fname.startswith(".") == False]

    for fname in fnames:
        with open(args.in_dir + fname, 'r') as f:
            text = f.read()
            lemmatized = lemmatize_text(text)
            with open(args.out_dir + fname, 'w') as o:
                o.write(lemmatized)
            print("Finished", fname)
