#!/usr/bin/env python

"""
parse and sample blurbs from the Blurb Genre Collection, 
compiled by U. Hamburg's Language Technology Group
"""

from argparse import ArgumentParser
import pandas as pd
from bs4 import BeautifulSoup

def parse_xml(soup):
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

    # a few entries don't have authors. not sure why, but we'll
    # filter them out
    blurbs = blurbs[blurbs['AUTHOR'] != '']
    blurbs = blurbs[~blurbs.duplicated()]

    return blurbs

def sample_blurbs(blurbs, n_samples=1000):
    sampled = blurbs.sample(n_samples)
    sampled = sampled.reset_index(drop=True)

    #generating file names
    sampled['FILE_NAME'] = [str(idx).zfill(4) + ".txt" for idx in sampled.index]

    return sampled

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file')
    parser.add_argument('--out_dir')
    parser.add_argument('--n_samples', type=int)
    parser.add_argument('--file_manifest')
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        data = f.read()
    
    soup = BeautifulSoup(data, 'lxml')
    blurbs = parse_xml(soup)
    sampled = sample_blurbs(blurbs, args.n_samples)

    for idx in sampled.index:
        fname = sampled.at[idx, 'FILE_NAME']
        with open(args.out_dir + fname, 'w') as f:
            f.write(sampled.at[idx, 'BLURB'])

    del sampled['BLURB']
    sampled.to_csv(args.file_manifest)

