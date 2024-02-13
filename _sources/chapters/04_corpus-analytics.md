---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell}
:tags: [remove-cell]
import os
import warnings

os.chdir("..")
warnings.filterwarnings('ignore')
```

Corpus Analytics
================

This chapter moves from an individual text to a collection of texts, or a
**corpus**. While computational methods may lead us to discover interesting
things about texts in isolation, they're often at their best when we use them
to study texts at scale. We'll do just that in this chapter: below, you'll
learn how to compile texts into a corpus and generate several metrics about
these texts. Doing so will enable us to observe similarities/differences across
the corpus at scale.

We'll also leave _Frankenstein_ behind and turn instead to Melanie Walsh's
[collection][collection] of ~380 obituaries from the _New York Times_.
"Obituary subjects," writes Walsh, "include academics, military generals,
artists, athletes, activities, politicians, and businesspeople – such as Ada
Lovelace, Ulysses Grant, Marilyn Monroe, Virginia Woolf, Jackie Robinson,
Marsha P. Johnson, Cesar Chavez, John F. Kennedy, Ray Kroc, and many more." 

[collection]: https://melaniewalsh.github.io/Intro-Cultural-Analytics/00-Datasets/00-Datasets.html

```{admonition} Learning Objectives
By the end of this chapter, you will be able to:

+ Compile texts into a corpus
+ Use a document-term matrix to represent relationships in the corpus
+ Generate metrics about texts in a corpus
+ Explain the difference between raw term metrics and weighted term scoring
```

Preliminaries
-------------

### Setup

Before we get into corpus construction, let's load the libraries we'll be using
throughout the chapter.

```{code-cell}
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
```

Initialize a path to the data directory as well.

```{code-cell}
indir = Path("data/session_two")
```

### File manifest

We'll also use a file manifest for this chapter. It contains metadata about the
corpus, and we'll also use it to keep track of our corpus construction.

```{code-cell}
manifest = pd.read_csv(indir.joinpath("manifest.csv"), index_col = 0)
manifest.info()
```

A brief plot of the years covered:

```{code-cell}
fig, ax = plt.subplots(figsize = (15, 5))
g = sns.histplot(x = 'year', bins = 30, data = manifest, ax = ax)
g.set(title = 'Obituaries per year', xlabel = 'Year', ylabel = 'Count');
```

```{tip}
Using a metadata sheet like this is a good habit to develop. Use it as a common
reference point for any processes you run on your data, and you'll metigate
major headaches stemming from undocumented projects. For more on this, see the
DataLab's [workshop on project organization and data documentation][projorg].

[projorg]: https://ucdavisdatalab.github.io/workshop_how-to-data-documentation
```

Compiling a Corpus
------------------

The obituaries are stored as individual files, which we need to load into
memory and combine into a corpus. Because cleaning can be labor- and
time-intensive, **these texts have already been cleaned (lemmatization
included).** You can find the script that performed this cleaning
[here][utils].

[utils]: https://github.com/ucdavisdatalab/workshop_getting_started_with_textual_data/tree/main/src/utils

Corpus compilation is fairly straightforward: we'll load everything in a `for`
loop. But the order of these texts is important, and this is where the file
manifest comes in. Using the manifest, we'll load the texts in the order
provided by the `file_name` column. Doing so ensures that the first index (`0`)
of our corpus corresponds to the first text, the second index (`1`) to the
second, and so on.

```{code-cell}
corpus = []
for title in manifest.index:
    fname = manifest.loc[title, 'file_name']
    path = indir.joinpath(f"input/{fname}")
    with path.open('r') as fin:
        corpus.append(fin.read())
```

Let's double-check that this has worked. First, we run an assertion to ensure
that the number of texts match the number of entries in our manifest.

```{code-cell}
assert len(corpus) == len(manifest), "Length's don't match!"
```

Second, let's inspect a few pieces of the texts.

```{code-cell}
for idx in range(3):
    print(corpus[idx][:25])
```

Looks great!

The Document-Term Matrix
------------------------

Before we switch into data exploration mode, we'll perform one last formatting
process on our corpus. Remember from the last chapter that much of text
analytics relies on **counts** and **contexts**. As with _Frankenstein_, one
thing we want to do is tally up all the words in each text. That produces one
kind of context – or rather, a few hundred different contexts: one for each
file we've loaded. But we also compiled these files into a corpus, the analysis
of which requires a different kind of context: a single one for all the texts.
That is, we need a way to relate texts _to each other_, instead of only
tracking word values inside a single text.

To do so, we'll build a **document-term matrix**, or **DTM**. A DTM is a matrix
that contains the frequencies of all terms in a corpus. Every row in this
matrix corresponds to a text, while every column corresponds to a term. For a
given text, we count the number of times that term appears and enter that
number in the column in question. We do this _even if_ the count is zero; key
to the way a DTM works is that it represents corpus-wide relationships between
texts, so it matters if a text does or doesn't contain a term.

Here's a toy example. Imagine three documents:

1. "I like cats. Do you?"
2. "I only like dogs. And you?"
3. "I like cats and dogs."

Transforming them into a DTM would yield:

```{code-cell}
example = [
    [1, 0, 1, 1, 0, 0, 1, 1],
    [1, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 0]
]
names = ['D1', 'D2', 'D3']
terms = ['i', 'only', 'like', 'cats', 'and', 'dogs', 'do', 'you']
example_dtm = pd.DataFrame(example, columns = terms, index = names)
example_dtm
```

Representing texts in this way is incredibly useful because it enables us to
discern similarities/differences in our corpus with ease. For example, all of
the above documents contain the words "i" and "like." Given that, if we
wanted to know what makes each document unique, we could ignore those two words
and focus on the rest of the values.

Now imagine doing this for thousands of words. What patterns might emerge?

```{margin} More on this
`CountVectorizer` accepts several arguments that modify its base functionality,
including arguments for applying some text cleaning steps. We won't use any of
these arguments because we've cleaned our texts already, but you can learn more
about them [here][cvdoc].

[cvdoc]: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
```

The `scikit-learn` library makes generating a DTM very easy. All we need to do
is initialize a `CountVectorizer` and fit it to our corpus. This does two
things: 1) it produces a series of different attributes that are useful for
corpus exploration; 2) it generates our DTM.

```{code-cell}
count_vectorizer = CountVectorizer()
vectorized = count_vectorizer.fit_transform(corpus)

print("Shape of our DTM:", vectorized.shape)
```

`CountVectorizer` returns a **sparse matrix**, or a matrix comprised mostly of
zeros. This matrix is formatted to be highly memory efficient, which is useful
when dealing with big datasets, but it's not very accessible for data
exploration. Since our corpus is relatively small, we'll convert this sparse
matrix to a `pandas` DataFrame.

We will also call our vectorizer's `.get_feature_names_out()` method. This
generates an array of all the tokens from all the obituaries. The order of this
array corresponds to the order of the DTM's columns. Accordingly, we assign
this array to the column names of our DTM. Finally, we assign the `name` column
values from our manifest to the index of our DTM.

```{code-cell}
dtm = pd.DataFrame(
    vectorized.toarray(),
    columns = count_vectorizer.get_feature_names_out(),
    index = manifest['name']
)
```

Raw Metrics
-----------

With the DTM made, it's time to use it to generate some metrics about each text
in the corpus. We use `pandas` data manipulations in conjunction with `NumPy`
to do so; the function below will make plots for us.

```{code-cell}
def plot_metrics(data, col, title = '', xlabel = ''):
    """Plot metrics with a histogram."""
    fig, ax = plt.subplots(figsize = (15, 5))
    g = sns.histplot(x = col, data = data)
    g.set(title = title, xlabel = xlabel, ylabel = 'Count');
```

### Documents

Here's an easy one: let's count the number of tokens in each text and assign
the result to a new column in our manifest.

```{code-cell}
manifest.loc[:, 'num_tokens'] = dtm.apply(sum, axis = 1).values
plot_metrics(manifest, 'num_tokens', title = 'Token counts', xlabel = 'Tokens')
```

We can also count the number of unique words, or **types**, in each text. Types
correspond to a text's vocabulary, whereas tokens correspond to the amount of
each of those types.

```{code-cell}
manifest.loc[:, 'num_types'] = dtm.apply(np.count_nonzero, axis = 1).values
plot_metrics(manifest, 'num_types', title = 'Type counts', xlabel = 'Types')
```

We can use tokens and types to generate a metric of **lexical diversity**.
There are a few such measures. We'll use the **type-token ratio (TTR)**, which
measures how much the vocabulary of a text varies over its tokens. It's a
simple metric: divide the number of types by the total number of tokens in a
text and normalize the result. A text with a TTR of 100, for example, would
never repeat a word.

```{code-cell}
manifest.loc[:, 'TTR'] = (manifest['num_types'] / manifest['num_tokens']) * 100
plot_metrics(manifest, 'TTR', title = 'Type-token ratio', xlabel = 'TTR')
```

What are examples of texts that typify the low, medium, and high range of these
metrics? The function below will pull up examples for each range.

```{margin} Code explanation
This function uses `qcut()` to bin metric values into a set of categories. The
output of `qcut()` is a Series. We match the index of this Series to our
DataFrame, pull out a random name from each bin, and print it.
```

```{code-cell}
def show_examples(data, metric, labels = ['low', 'medium', 'high']):
    cuts = pd.qcut(data[metric], len(labels), labels = labels)
    for label in labels:
        idxes = cuts[cuts==label].index
        ex = data[data.index.isin(idxes)].sample()
        print(f"+ {label}: {ex['name'].item()} ({ex[metric].item():0.2f})")

for metric in ('num_tokens', 'num_types', 'TTR'):
    print("Examples for", metric)
    show_examples(manifest, metric)
    print("\n")
```


### Terms

Let's move on to terms. Here are the top-five most frequent terms in the
corpus:

```{code-cell}
summed = pd.DataFrame(dtm.sum(), columns = ('count', ))
summed.sort_values('count', ascending = False, inplace = True)
summed.head(5)
```

And here are the bottom five:

```{code-cell}
summed.tail(5)
```

Though there are likely to be quite a few one-count terms. We refer to them as
**hapax legomena** (Greek for "only said once"). How many are in our corpus
altogether?

```{code-cell}
hapaxes = summed[summed['count'] == 1]
print(f"{len(hapaxes)} hapaxes ({len(hapaxes) / len(dtm.T) * 100:0.2f}%)")
```

If we plot our term counts, we'll see a familiar pattern: our term distribution
is Zipfian.

```{code-cell}
fig, ax = plt.subplots(figsize = (15, 5))
g = sns.lineplot(x = summed.index, y = 'count', data = summed)
g.set(title = 'Term distribution', xlabel = 'Terms', xticks = []);
```

This distribution has a few consequences. It suggests, for example, that we
might have more cleaning to do in terms of our stopwords: perhaps our list
hasn't accounted for some words, and if we don't remove them, we might have
trouble identifying more unique aspects of each text in our corpus. But on the
other hand, all of those hapaxes are a little _too_ unique: one option would be
to drop them entirely.

But dropping our hapaxes might lose out on some specificity in our texts. And
there are some highly frequent words that we want to retain ("year" and "make"
for example). So what to do? How can we reduce the influence of highly frequent
terms without removing them, and how can we take into account rare words at the
same time?

Weighted Metrics
----------------

The answer is to **weight** our terms, doing so in a way that lessens the
impact of high frequency terms and boosts that of rare ones. The most popular
weighting method is **tf-idf, or term frequency--inverse document frequency,
scoring**. A tf-idf score is a measure of term specificity in the context of a
given document. It is the product of a term's frequency in that document and
the number of documents in which that term appears. By offsetting terms that
appear across many documents, tf-idf pushes down the scores of common terms and
boosts the scores of rarer ones.

A tf-idf score is expressed

```{margin} The idf score
We calculate the inverse document frequency score with

$$
idf_{i} = log(\frac{n}{df_{i}})
$$

Where the score for term $i$ is the log of the total number of total documents
($n$) over the number of documents that contain $i$.
```

$$
score_{ij} = tf_{ij} \cdot idf_{i}
$$

Where, for term $i$ and document $j$, the score is the term frequency
($tf_{ij}$) for $i$ multiplied by its inverse document frequency ($idf_{i}$).
The higher the score, the more specific a term is to a given document.

We don't need to implement any of this math ourselves. `scikit-learn` has a
`TfidfVectorizer` object, which works just like `CountVectorizer`, but instead
of producing a DTM of raw term counts, it produces one with tf-idf scores.

```{code-cell}
tfidf_vectorizer = TfidfVectorizer()
vectorized_tfidf = tfidf_vectorizer.fit_transform(corpus)
tfidf = pd.DataFrame(
    vectorized_tfidf.toarray(),
    columns = tfidf_vectorizer.get_feature_names_out(),
    index = manifest['name']
)
```

To see the difference tf-idf scores make, let's compare raw counts and tf-idf
scores for three texts.

```{code-cell}
def counts_vs_tfidf(dtm, tfidf, idx, n_terms = 10):
    """Compare raw document counts with tf-idf scores."""
    c = dtm.loc[idx].nlargest(n_terms)
    t = tfidf.loc[idx].nlargest(n_terms)
    df = pd.DataFrame({
        'count_term': c.index, 'count': c.values,
        'tfidf_term': t.index, 'tfidf': t.values
    })

    return df

for name in manifest['name'].sample(3):
    comparison = counts_vs_tfidf(dtm, tfidf, name)
    print("Person:", name)
    display(comparison)
```

There are a few things to note here. First, names often have the largest value,
regardless of our metric. This makes intuitive sense for our corpus: we're
examining obituaries, where the sole subject of each text may be referred to
many times. A tf-idf score will actually reinforce this effect, since names are
often specific to the obituary. We can see this especially from the shifts taht
take place in the rest of the rankings. Whereas raw counts will often refer to
more general nouns and verbs, tf-idf scores home in on other people with whom
the present person might've been associated, places that person might've
visited, and even things that particular person is known for. Broadly speaking,
tf-idf scores give us a more situated sense of the person in question.

To see this in more detail, let's look at a single obituary in the context of
the entire corpus. We'll compare the raw counts and tf-idf scores of this
obituary to the mean counts and scores of the corpus.

```{code-cell}
kahlo = counts_vs_tfidf(dtm, tfidf, 'Frida Kahlo', 15)
kahlo.loc[:, 'corpus_count'] = dtm[kahlo['count_term']].mean().values
kahlo.loc[:, 'corpus_tfidf'] = tfidf[kahlo['tfidf_term']].mean().values

cols = ['count_term', 'count', 'corpus_count', 'tfidf_term', 'tfidf', 'corpus_tfidf']
kahlo = kahlo[cols]
kahlo
```

In the next chapter we'll take this one step further. There, we'll use
corpus-wide comparisons of tf-idf scores to identify similar obituaries.

With that in mind, we'll save our tf-idf DTM and end here.

```{code-cell}
outdir = indir.joinpath("output")
tfidf.to_csv(outdir.joinpath("tfidf_scores.csv"))
```
