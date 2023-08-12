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

os.chdir("..")
```

Clustering and Classification
=============================

This chapter directly extends the last. Chapter 4 showed you how to build a
corpus from a collection of text files and product different metrics about
them. For the most part these metrics hewed either toward the level of
documents or that of words; we didn't discuss ways of weaving these two aspects
of our corpus together. In this chapter we'll learn how to do so by identifying
similar documents with a special measure, cosine similarity. With this measure,
we'll be able to cluster our corpus into distinct groups, which may in turn
tell us something about its overall shape, trends, patterns, etc.

The applications of this technique are broad. Search engines use cosine
similarity to identify relevant results to queries; literature scholars consult
it to investigate authorship and topicality. Below, we'll walk through the
concepts that underlie cosine similarity and show a few examples of how to
explore and interpret its results.

```{admonition} Learning Objectives
By the end of this chapter, you will be able to:

+ Rank documents by similarity, using cosine similarity
+ Project documents into a feature space to visualize their similarities
+ Cluster documents by their similarities
```

Preliminaries
-------------

As before, we'll first load some libraries.

```{code-cell}
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
```

And we will load our data as well.

```{code-cell}
dtm = pd.read_csv("data/session_two/output/tfidf_scores.csv", index_col = 0)
```

The Vector Space Concept
------------------------

Measuring similarities requires a conceptual leap. Imagine projecting every
document in our corpus into a space. In this space, similar documents have
similar orientations to a point of origin (in this case, the origin point of XY
axes), while dissimilar ones have different orientations. Guiding these
orientations are the values for each term in a document. Here, _space is a
metaphor for semantics_.

Here is a toy example to demonstrate this idea. Imagine two documents, each
with a value for two terms.

```{code-cell}
toy = [[18, 7], [4, 12]]
toy = pd.DataFrame(toy, columns = ('page', 'screen'), index = ('d1', 'd2'))
toy
```

Since there are only two terms here, we can plot them using XY coordinates.

```{code-cell}
def quiver_plot(dtm, col1, col2):
    """Create a quiver plot from a DTM."""
    origin = [[0, 0], [0, 0]]
    fig, ax = plt.subplots(figsize = (8, 8))
    ax.quiver(*origin, dtm[col1], dtm[col2], scale = 1, units = 'xy')
    ax.set(
        xlim = (0, 20), ylim = (0, 20)
        , xticks = range(0, 21), yticks = range(0, 21)
        , xlabel = col1, ylabel = col2
    )
    for doc in toy.index:
        ax.text(
            dtm.loc[doc, col1], dtm.loc[doc, col2] + 0.5
            , doc, va = 'top', ha = 'center'
        )
    plt.show()

quiver_plot(toy, 'page', 'screen')
```

The important part of this plot is the angle created by the two documents.
Right now, this angle is fairly wide, as the differences between the two counts
in each document are rough inverses of one another. 

But if we change the counts in our first document to be more like the second...

```{code-cell}
toy2 = [[10, 18], [4, 12]]
toy2 = pd.DataFrame(toy2, columns = ('page', 'screen'), index = ('d1', 'd2'))
toy2
```

And re-plot:

```{code-cell}
quiver_plot(toy2, 'page', 'screen')
```

We will shrink the angle considerably. The first document is now more like the
second. Importantly, this similarity is relatively impervious to the actual
value of each word. The counts in each document are different, but the _overall
relationship_ between the values for each word is now similar: both documents
have more instances of "screen" than "page," and our method of projecting them
into this space, or what we call a **vector space**, captures this.

Cosine Similarity
-----------------

**Cosine similarity** describes this difference in angles. This score measures
the inner product space between two vectors, that is, the amount of space
created by divergences in the attributes of each vector. Typically, the score
is expressed within a bounded interval of [0, 1], where 0 stands for a right
angle between two vectors (no similarity) and 1 measures vectors that
completely overlap (total similarity).

Formally, cosine similarity is expressed

$$
similarity(A, B) = \frac{A \cdot B}{\Vert A \Vert \cdot \Vert B \Vert}
$$

But we don't need to worry about implementing this ourselves. `scikit-learn`
has a one-liner for this. It outputs a square matrix of values, where each cell
is the cosine similarity measure of the intersection between a column and a
row.

Here's the first example:

```{code-cell}
cosine_similarity(toy)
```

And here's the second:

```{code-cell}
cosine_similarity(toy2)
```

Of the two examples, which one has more similar documents? The second one: the
cosine similarity score is 0.98, far higher than 0.64.

### Generating scores

For our purposes, the most powerful part of cosine similarity is that it takes
into account many dimensions, not just two. That means we can send it our
tf-idf scores, which have many dimensions:

```{code-cell}
print("Number of dimensions per text:", len(dtm.columns))
```

We're working in very high-dimensional space! While this might be hard to
conceptualize (and indeed impossible to visualize), `scikit-learn` can do the
heavy lifting; we'll just look at the results. Below, we use
`cosine_similarity()` to compute scores for comparisons across every document
combination in the corpus.

```{code-cell}
sims = cosine_similarity(dtm)
sims = pd.DataFrame(sims, columns = dtm.index, index = dtm.index)
sims.iloc[:5, :5]
```

Now we're able to identify the most and least similar texts to a given
obituary. Below, we find most similar ones. When making a query, we need the
second index position since the highest score will always be the obituary's
similarity to itself.

```{margin} Code explanation
For every person, get their cosine similarity scores. Find the largest scores
and use `.take()` to pull out the second-largest one.
```

```{code-cell}
people = ('Ada Lovelace', 'Henrietta Lacks', 'FDR', 'Miles Davis')
for person in people:
    scores = sims.loc[person]
    most = scores.nlargest().take([1])
    print(f"For {person}: {most.index.item()} ({most.item():0.3f})")
```

### Visualizing scores

It would be helpful to have a bird's eye view of the corpus as well. But
remember that our similarities have hundreds of attributes – far more than it's
possible to visualize. We'll need to reduce the dimensionality of this data,
decomposing the original values into a set of two for each text. We'll use
**t-distributed stochastic neighbor embedding** (t-SNE) to do so. The method
takes a matrix of multi-dimensional data and decomposes it into a lower set of
dimensions, which represent the most important features of the input.

`scikit-learn` has a built-in for t-SNE:

```{margin} Want to dive into the weeds?
See the `scikit-learn` [documentation][tsne] for an overview of t-SNE and
citations to the papers that first introduced it.

[tsne]: https://scikit-learn.org/stable/modules/manifold.html#t-sne
```

```{code-cell}
reducer = TSNE(
    n_components = 2
    , learning_rate = 'auto'
    , init = 'random'
    , angle = 0.35
    , random_state = 357
    , n_jobs = -1
)
reduced = reducer.fit_transform(sims)
```

With our data transformed, we'll store the two dimensions in a DataFrame.

```{code-cell}
vis = pd.DataFrame({'x': reduced[:,0], 'y': reduced[:,1], 'label': sims.index})
```

Time to visualize. As with past chapters, we define a function to make multiple
graphs.

```{code-cell}
def sim_plot(data, hue = None, labels = None, n_colors = 3):
    """Create a scatterplot and optionally color/label its points."""
    fig, ax = plt.subplots(figsize = (10, 10))
    pal = sns.color_palette('colorblind', n_colors = n_colors) if hue else None
    g = sns.scatterplot(
        x = 'x', y = 'y'
        , hue = hue, palette = pal, alpha = 0.8
        , data = data, ax = ax
    )
    g.set(xticks = [], yticks = [], xlabel = 'Dim. 1', ylabel = 'Dim. 2')

    if labels:
        to_label = data[data['label'].isin(labels)]
        to_label[['x', 'y', 'label']].apply(lambda x: g.text(*x), axis = 1)

    plt.show();

sim_plot(vis, labels = people)
```

As with the toy example above, here, space has semantic value. Points that
appear closer together in the graph have more similarities between them than
those that are farther apart. The following pair of people illustrates this
well.

```{margin} A caveat
It's important to remember that the transformation from high-dimensional space
to this 2D plot is a lossy one. Distortions will result form it, and having a
good sense of what kind of distortions you might expect to see with t-SNE will
help you interpret it. This [interactive dashboard][dash] offers a really nice
overview of these considerations.

[dash]: https://distill.pub/2016/misread-tsne/
```

```{code-cell}
roosevelts = ('FDR', 'Eleanor Roosevelt')
sim_plot(vis, labels = roosevelts)
```

Quite close!

Clustering by Similarity
------------------------

The last topic we'll cover in this chapter involves dividing up the feature
space of our data so as to cluster the texts in our corpus into broad
categories. This will provide a frame for inspecting general trends in the
data, which may in turn help us understand the specificity of individual texts.
For example, clustering might allow us to find various subgenres of obituaries.
It could also tell us something about how the _New York Times_ writes about
particular kinds of people (their vocations, backgrounds, etc.). It may even
indicate something about how the style of the obituary has changed over time.

But how to do it? Visual inspection of the graphs would be one way, and there
are a few cluster-like partitions in the output above. But those cues can be
misleading. It would be better to use methods that can partition our feature
space automatically.

Below, we'll use **hierarchical clustering** to do this. In hierarchical
clustering, each data point is treated as an individual cluster before being
merged with its most similar neighbor. Then, the two points are merged with
another cluster and so on. Merging continues in an upward fashion, growing the
size of each cluster until the process reaches a predetermined number of
clusters.

```{margin} How many clusters?
We can't go into detail about this question. It's a complex topic, and much
depends on a) your data; and b) what you want to know about it. For now, we'll
stick with a general approach, which you can nuance later on.
```

Once again, `scikit-learn` has functionality for clustering. Below, we fit an
`AgglomerativeClustering` object to our similarities.

```{code-cell}
k = 3
agg = AgglomerativeClustering(n_clusters = k)
agg.fit(sims)
```

Cluster labels are stored in the `labels_` attribute of this object. We'll
associate them with our visualization data to color points by cluster.

```{code-cell}
vis.loc[:, 'cluster'] = agg.labels_
sim_plot(vis, hue = 'cluster', labels = people) 
```

There are a few incursions of one cluster into another, but for the most part
this looks pretty coherent. Let's grab some examples of each cluster and show
them.

```{margin} Code explanation
First, group the cluster data. For each groupe, sample `n_sample` entries.
Print each label of the sample results.
```

```{code-cell}
n_sample = 5
for group in vis.groupby('cluster'):
    cluster, subset = group
    print(f"Cluster: {cluster}\n----------")
    samp = subset.sample(n_sample)
    for label in samp['label']:
        print(label)
    print("\n")
```

Looking over these results turns up an interesting pattern: they're tied to
profession.

1. Cluster 0: Entertainers and performers (Billie Holiday, Marilyn Monroe,
   etc.)
2. Cluster 1: Political figures (Ghandi, Adlai Stevenson, etc.)
3. Cluster 2: Artists, activists, and other public figures (Helen Keller,
   Joseph Pulitzer, Jackie Robinson)

From an initial glance, our similarities seem to suggest that a _person's
profession indicates something about the type of obituary they have_.

Of course, there are exceptions. Mao Tse Tung (Mao Zedong) is in the third
cluster. Though he was a poet, he was best known as a politician; the second
cluster might make for a better home. Tolstoy, who _is_ in the second cluster,
presents an interesting test case. While known primarily as a novelist, he was
from an aristocratic family, and his stance towards non-violence has been
influential for a number of political figures. Should he therefore be in the
third cluster, or is he in his rightful place here?

```{code-cell}
unexpected = ('Mao Tse Tung', 'Tolstoy')
sim_plot(vis, hue = 'cluster', labels = unexpected)
```

The question for both exceptions is: why? What is it about these two people's
obituaries that pushes them outside the cluster we'd expect them to be in? Is
it a particular set of words? Lexical diversity? Something else?

Those questions will take us beyond the scope of this chapter. For now, it's
enough to know that text similarity prompts them. More, we can explore and
analyze corpora using this metric. We'll end, then, with this preliminary
interpretation of our clusters, which later work will need to corroborate or
challenge.

