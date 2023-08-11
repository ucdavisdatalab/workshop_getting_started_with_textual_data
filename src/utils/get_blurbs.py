#!/usr/bin/env python

import argparse
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

def parse_xml(soup):
    """Parse book XML and extract its metadata."""
    blurbs = []
    for book in soup.find_all('book'):
        entry = {
            'author': book.find('author').text,
            'title': book.find('title').text,
            'blurb': book.find('title').next_sibling,
            'genre': book.find('d0').text,
            'pub_date': book.find('published').text,
            'isbn': book.find('isbn').text
        }
        blurbs.append(entry)

    blurbs = pd.DataFrame(blurbs)
    blurbs['pub_date'] = pd.to_datetime(blurbs['pub_date'])
    blurbs['isbn'] = blurbs['isbn'].astype(int)

    blurbs = blurbs[blurbs['author'] != '']
    blurbs = blurbs[~blurbs.duplicated()]

    return blurbs

def sample_blurbs(blurbs: pd.DataFrame, n_samples: int=1000) -> pd.DataFrame:
    """Sample blurbs."""
    sampled = blurbs.sample(n_samples)
    sampled = sampled.reset_index(drop=True)

    sampled.loc[:, 'file_name'] = [str(idx).zfill(4) for idx in sampled.index]

    return sampled

def main(args: argparse.Namespace) -> None:
    """Run the script."""
    with args.file.open('r') as fin:
        data = fin.read()
    
    soup = BeautifulSoup(data, features='lxml')
    blurbs = parse_xml(soup)
    sampled = sample_blurbs(blurbs, args.n_samples)

    for idx in sampled.index:
        fname = sampled.at[idx, 'file_name']
        outpath = args.outdir.joinpath(fname).with_suffix(".txt")
        with outpath.open('w') as fout:
            fout.write(sampled.at[idx, 'blurb'])

    del sampled['blurb']
    sampled.to_csv(args.file_manifest)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parse and sample blurbs from the Blurb Genre Collection"
    )
    parser.add_argument('--file', type=Path, help="Path to file")
    parser.add_argument('--outdir', type=Path, help="Output directory")
    parser.add_argument('--n_samples', type=int, help="Number of samples")
    parser.add_argument('--file_manifest', type=Path, help="File manifest")
    args = parser.parse_args()
    main(args)


