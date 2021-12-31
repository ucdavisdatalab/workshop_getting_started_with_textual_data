#! /usr/bin/env python

"""
Helper script to quickly clean files
"""

import re

with open("../data/voyant_stoplist.txt", 'r') as f:
	stopwords = f.read().split()

def remove_extra_chars(doc):
	"""
	Remove anything that isn't a letter
	"""
	doc = re.sub(r"[-]|[—]|[_]", " ", doc)
	doc = re.sub(r"[^\w\s]", "", doc)
	doc = re.sub(r"\s+", " ", doc)
	return doc

def apply_stopwords(doc):
	"""
	Stop out a string
	"""
	doc = doc.split()
	doc = [token for token in doc if token not in stopwords and len(token) > 2]
	return ' '.join(doc)

def clean(doc):
	"""
	Clean a string; returns full strings (not bags-of-words)
	"""
	doc = remove_extra_chars(doc.lower())
	doc = apply_stopwords(doc)
	return doc
