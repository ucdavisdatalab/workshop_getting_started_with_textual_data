#!/usr/bin/env python

"""
parse and sample blurbs from the Blurb Genre collection, 
compiled by U. Hamburg's Language Technology Group
"""

import pandas as pd
from bs4 import BeautifulSoup

print("Enter filepath to data:")
inpath = input()

print("\nEnter filepath to output:")
outdir = input()

with open(inpath + "BlurbGenreCollection_EN_train.txt", 'r') as f:
    data = f.read()
    soup = BeautifulSoup(data, 'lxml')

# roll through each book and get metadata for a file manifest
blurbs = []
for book in soup.find_all('book'):
    entry = {
        'AUTHOR': book.find('author').text,
        'TITLE': book.find('title').text,
        'BLURB': book.find('title').next_sibling,
        'GENRE': book.find('d0').text,
        'PUB_DATE': book.find('published').text,
        'ISBN': book.find('isbn').text
    }
    blurbs.append(entry)

# set some datatypes
blurbs = pd.DataFrame(blurbs)
blurbs['PUB_DATE'] = pd.to_datetime(blurbs['PUB_DATE'])
blurbs['ISBN'] = blurbs['ISBN'].astype(int)

# a few entries don't have authors. not sure why, but we'll filter them out
blurbs = blurbs[blurbs['AUTHOR'] != '']
blurbs = blurbs[~blurbs.duplicated()]

# sample
n_samples = 1000
sampled = blurbs.sample(n_samples)
sampled = sampled.reset_index(drop=True)

# create filenames
sampled['FILE_NAME'] = [str(idx).zfill(3) + ".txt" for idx in sampled.index]

# save
for idx in sampled.index:
    fname = sampled.at[idx, 'FILE_NAME']
    with open(outdir + fname, 'w') as f:
        f.write(sampled.at[idx, 'BLURB'])

sampled.to_csv(outdir + "manifest.csv")
