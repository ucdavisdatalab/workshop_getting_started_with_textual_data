#! /usr/bin/env python

"""
Lemmatize a directory of texts. Assumes that there is a file manifest in that 
directory called "manifest" that contains a column of filenames, titled `FILE_NAME`
"""

import pandas as pd
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

print("Enter an input directory path")
indir = input()

print("\nEnter an output directory path")
outdir = input()

print(f"\nLoading files from {indir} and outputting them to {outdir}\n")
manifest = pd.read_csv(indir + "manifest.csv", index_col=0)
tokenizer = TreebankWordTokenizer()
detokenizer = TreebankWordDetokenizer()
lemmatizer = WordNetLemmatizer()

def convert_tag(tag):
	"""
	Convert TreeBank tags to WordNet tags
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
	Lemmatize using WordNet tags (when available)
	"""
	if tag != '':
		return lemmatizer.lemmatize(word, pos=tag)
	else:
		return lemmatizer.lemmatize(word)

def lemmatize_text(doc):
	"""
	Tokenize a text, tag it, and then use those tags to lemmatize the text. 
	Return a detokenized version of the text (for teaching purposes we want 
	participants to tokenize texts on their own)
	"""
	bow = tokenizer.tokenize(doc)
	tagged = nltk.pos_tag(bow)
	tagged = [(i[0], convert_tag(i[1])) for i in tagged]
	lemmatized = [lemmatize_word(i[0], i[1]) for i in tagged]
	return detokenizer.detokenize(lemmatized)

for fname in manifest['FILE_NAME']:
	path = indir + fname
	with open(path, 'r') as f:
		text = f.read()
		lemmatized = lemmatize_text(text)
		with open(outdir + fname, 'w') as ff:
			ff.write(lemmatized)
	print("Finished", fname)
