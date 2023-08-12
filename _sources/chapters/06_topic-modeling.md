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

Topic Modeling
==============

This final chapter follows from the previous chapter's use of cosine
similarity. The latter used this metric to cluster obituaries into broad
categories based on what those obituaries were about. Similarly, in this
chapter we'll use **topic modeling** to identify the thematic content of a
corpus and, on this basis, associate themes with individual documents.

As we'll discuss below, human interpretation plays a key role in this process:
topic models produce textual structures, but it's on us to give those
structures meaning. Doing so is an iterative process, in which we **fine tune**
various aspects of a model to effectively represent our corpus. This chapter
will show you how to build a model, how to appraise it, and how to start
iterating through the process of fine tuning to produce a model that best
serves your research questions.

To do so, we'll use a corpus of book blurbs sampled from the U. Hamburg
Language Technology Group's [Blurb Genre Collection][bgc]. The collection
contains ~92,000 blurbs from Penguin Random House, ranging from Colson
Whitehead's books to steamy supermarket romances and self-help manuals. We'll
use just 1,500 – not so much that we'd be stuck waiting for hours for models to
train, but enough to get a broad sense of different topics among the blurbs.

[bgc]: https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html

```{admonition} Learning Objectives
By the end of this chapter, you will be able to:

+ Explain what a topic model is, what it represents, and how to use one to
  explore a corpus
+ Build a topic model
+ Use two scoring metrics, perplexity and coherence, to appraise the quality of
  a model
+ Understand how to improve a model by fine tuning its number of topics and its
  hyperparameters
```

Topic Modeling: Introduction
----------------------------

```{margin} Citations
This description of topic modeling is influenced by Ted Underwood's very
effective [explanation][underwood]. This [review article][blei], from David M.
Blei (one of the original developers of the method), is also quite helpful.

[underwood]: https://tedunderwood.com/2012/04/07/topic-modeling-made-just-simple-enough/
[blei]: http://www.cs.columbia.edu/~blei/papers/Blei2012.pdf
```

There are a few different flavors of topic models. We'll be using the most
popular one, a **latent Dirichlet allocation**, or LDA, model. It involves two
assumptions: 1) documents are comprised of a mixture of topics; 2) topics are
comprised of a mixture of words. An LDA model represents these mixtures in
terms of probability distributions: a given passage, with a given set of words,
is more or less likely to be about a particular topic, which is in turn more or
less likely to be made up of a certain grouping of words.

We initialize a model by predefining how many topics we think it should find.
When the model begins training, it randomly guesses which words are most
associated with which topic. But over the course of its training, it will start
to keep track of the probabilities of recurrent word collocations: "river",
"bank," and "water," for example, might keep showing up together. This suggests
some coherence, a possible grouping of words. A second topic, on the other
hand, might have words like "money," "bank," and "robber." The challenge here
is that words belong to multiple topics. In this instance, given a single
instance of "bank," it could be in either the first or second topic. Given
this, how does the model tell which topic a document containing the word "bank"
is more strongly associated with?

It does two things. First, the model tracks how often "bank" appears with its
various collocates in the corpus. If "bank" is generally more likely to appear
with "river" and "water" than "money" and "robber", this weights the
probability that this particular instance of "bank" belongs to the first topic.
To put a check on this weighting, the model also tracks how often collocates of
"bank" appear in the document in question. If, in this document, "river" and
"water" appear more often than "robber" and "money," then that will weight this
instance of "bank" even further toward the first topic, not the second.

Using these weightings as a basis, the model assigns a probability score for a
document's association with these two topics. This assignment will also inform
the overall probability distribution of topics to words, which will then inform
further document-topic associations, and so on. Over the course of this
process, topics become more consistent and focused and their associations with
documents become stronger and weaker, as appropriate.

Here's the formula that summarizes this process. Given topic $T$, word $W$, and
document $D$, we determine the probability of $W$ belonging to $T$ with:

```{margin} Those two other letters...
These represent hyperparameters, which we'll discuss below.
```

$$
P(T|W,D) = \frac{\text{# of $W$ in } T + \eta_W}{\text{total tokens in } T + \eta}
\cdot (\text{# words in $D$ that belong to } T + \alpha)
$$

Preliminaries
-------------

Before we begin building a model, we'll load the libraries we need and our
data. As we've done in past chapters, we use a file manifest to keep track of
things.

```{code-cell}
from pathlib import Path
import numpy as np
import pandas as pd
import tomotopy as tp
from tomotopy.utils import Corpus
from tomotopy.coherence import Coherence
import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis
```

Our input directory:

```{code-cell}
indir = Path("data/session_three")
```

And our manifest:

```{code-cell}
manifest = pd.read_csv(indir.joinpath("manifest.csv"), index_col = 0)
manifest.loc[:, 'year'] = pd.to_datetime(manifest['pub_date']).dt.year
manifest.info()
```

A small snapshot of its contents:

```{code-cell}
print(f"Number of blurbs: {len(manifest)}")
print(f"Pub dates: {manifest['year'].min()} -- {manifest['year'].max()}")
print(f"Genres: {', '.join(manifest['genre'].unique())}")
```

Building a Topic Model
----------------------

```{margin} You may be asking...
...why not use `scikit-learn`? Well, there's a [reported bug][bug] in this
implementation that has to do with a key metric we'll be using later on.

[bug]: https://github.com/scikit-learn/scikit-learn/issues/6777
```

With our preliminary work done, we're ready to build a topic model. There are
numerous implementations of LDA modeling available, ranging from the command
line utility, [MALLET][mallet], to built-in APIs offered by both `gensim` and
`scikit-learn`. We will be using `tomotopy`, a Python wrapper built around the
C++ topic modeling too, Tomato. Its API is fairly intuitive and comes with a
lot of options, which we'll leverage to build the best possible model for our
corpus.

[mallet]: https://mimno.github.io/Mallet/

### Initializing a corpus

Before we build the model, we need to load the data on which it will be
trained. We use `Corpus` to do so. Be sure to split each file into a list of
tokens before adding it to this object.

```{code-cell}
corpus = Corpus()
for fname in manifest['file_name']:
    path = indir.joinpath(f"input/{fname}")
    with path.open('r') as fin:
        doc = fin.read()
        corpus.add_doc(doc.split())
```

### Initializing a model

To initialize a model with `tomotopy`, all we need is the corpus from above and
the number of topics the model will generate. Determining how many topics to
use is a matter of some debate and complexity, which you'll learn more about
below. For now, just pick a small number. We'll also set a set for
reproducibility.

```{code-cell}
seed = 357
model = tp.LDAModel(k = 5, corpus = corpus, seed = seed)
```

### Training a model

Our model is now ready to be trained. Under the hood, this happens in an
iterative fashion, so we need to set the total number of iterations for the
training. With that set, it's simply a matter of calling `.train()`.

```{margin} Number of iterations
Deciding on the number of iterations to use takes some tweaking. You can assume
that the number set here will allow a model to properly [converge][converge]
for this data.

[converge]: https://machine-learning.paperspace.com/wiki/convergence
```

```{code-cell}
iters = 1000
model.train(iter = iters)
```

### Inspecting the results

Let's look at the trained model. For each topic, we can get the words that are
most associated with it. The accompanying score is the probability of that word
appearing in the topic.

```{code-cell}
def top_words(model, k):
    """Print the top words for topic k in a model."""
    top_words = model.get_topic_words(topic_id = k, top_n = 5)
    top_words = [f"{word} ({score:0.4f}%)" for (word, score) in top_words]
    print(f"Topic {k}: {', '.join(top_words)}")

for i in range(model.k):
    top_words(model, i)
```

These results make intuitive sense: we're dealing with several hundred book
blurbs, so we'd expect to see words like "book," "reader," and "new."

The `.get_topic_dist()` method performs a similar function, but for a document.

```{code-cell}
def doc_topic_dist(model, idx):
    """Print the topic distribution for a document."""
    topics = model.docs[idx].get_topic_dist()
    for idx, prob in enumerate(topics):
        print(f"+ Topic #{idx}: {prob:0.2f}%")

random_title = manifest.sample().index.item()
doc_topic_dist(model, random_title)
```

`tomotopy` also offers some shorthand to produce the top topics for a document.
Below, we sample from our manifest, send the indexes to our model, and retrieve
top topics.

```{code-cell}
sampled_titles = manifest.sample(5).index
for idx in sampled_titles:
    top_topics = model.docs[idx].get_topics(top_n = 1)
    topic, score = top_topics[0]
    print(f"{manifest.loc[idx, 'title']}: #{topic} ({score:0.2f}%)")
```

It's possible to get even more granular. Every word in a document in a document
has its own associated topic, which will change depending on the document. This
is about as close to context-sensitive semantics as we can get with this
method.

```{code-cell}
doc = model.docs[random_title]
word_to_topic = list(zip(doc, doc.topics))
for word in range(10):
    word, topic = word_to_topic[word]
    print(f"+ {word} ({topic})")
```

Let's zoom out to the level of the corpus and retrieve the topic probability
distribution for each document. In the literature, this is called the
**theta**. More informally, we'll refer to it as the **document-topic matrix**.

```{code-cell}
def get_theta(model, labels):
    """Get the theta matrix from a model."""
    theta = np.stack([doc.get_topic_dist() for doc in model.docs])
    theta = pd.DataFrame(theta, index = labels)

    return theta

theta = get_theta(model, manifest['title'])
theta
```

It's often helpful to know how large each topic is. There's a caveat here,
however, in that each word in the model technically belongs to each topic, so
it's somewhat of a heuristic to say that a topic's size is $n$ words.
`tomotopy` derives the output below by multiplying each column of the theta
matrix by the document lengths in the corpus. It then sums the results for each
topic.

```{code-cell}
topic_sizes = model.get_count_by_topics()
print("Number of words per topic:")
for i in range(model.k):
    print(f"+ Topic #{i}: {topic_sizes[i]} words")
```

Finally, using the `num_words` attribute we can express this as percentages:

```{code-cell}
print("Topic proportion across the corpus:")
for i in range(model.k):
    print(f"+ Topic #{i}: {topic_sizes[i] / model.num_words:0.2f}%")
```

Fine Tuning: The Basics
-----------------------

Everything is working so far. But our topics are extremely general. More, their
total proportions across the corpus are relatively homogeneous. This may be an
indicator that our model has not been fitted to our corpus particularly well.

Looking at the word-by-word topic distributions for a document from above shows
this well. Below, we once again print those out, but we'll add the top words
for each topic in general.

```{code-cell}
for word in range(5):
    word, topic = word_to_topic[word]
    print(f"Word: {word}")
    top_words(model, topic)
```

It would appear that the actual words in a document do not really match the top
words for its associated topic. This suggests that we need to make adjustments
to the way we initialize our model so that it better reflects the specifics of
our corpus.

But there are several different parameters to adjust. So what should we change?

### Number of topics

An easy answer would be the number of topics. If, as above, your topics seem
too general, it may be because you've set too small a number of topics for the
model. Let's set a higher number of topics for our model and see what changes.

```{code-cell}
model10 = tp.LDAModel(k = 10, corpus = corpus, seed = seed)
model10.train(iter = iters)

for k in range(model10.k):
    top_words(model10, k)
```

That looks better! Adding more topics spreads out the word distributions. Given
that, what if we increased the number of topics even more?

```{code-cell}
:tags: ["output_scroll"]
model30 = tp.LDAModel(k = 30, corpus = corpus, seed = seed)
model30.train(iter = iters)

for k in range(model30.k):
    top_words(model30, k)
```

This also looks pretty solid. The two models appear to share topics, but the
second model, which has a higher number of topics, includes a wider range of
words in the top word distribution. While all that seems well and good, we
don't yet have a way to determine whether an increase in the number of topics
will always produce more interpretable results. At some point, we might start
splitting hairs. In fact, we can already see this beginning to happen in a few
instances in the second model.

So the question is, what is an ideal number of topics?

One way to approach this question would be to run through a range of different
topic sizes and inspect the results for each. In some cases, it can be
perfectly valid to pick the number of topics that's most interpretable for you
and the questions you have about your corpus. But there are also a few metrics
available that will assess the quality of a given model in terms of the
underlying data it represents. Sometimes these metrics lead to models that
aren't quite as interpretable, but they also help us make a more empirically
grounded assessment of the resultant topics.

### Perplexity

The first of these measures is **perplexity**. In text analysis, we use
perplexity scoring to evaluate how well a model predicts an sample sequence of
words. Essentially, it measures how "surprised" a model is by that sequence.
The lower the perplexity, the more your model is capable of mapping predictions
against the data it was trained on.

```{margin} More on perplexity
If you'd like to read more about perplexity, [this post][post] offers a good
walkthrough of the concept and its application to the kind of work we're doing
here.

[post]: https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3
```

When you train a `tomotopy` model object, the model records a perplexity score
for the training run.

```{code-cell}
for m in (model10, model30):
    print(f"Perplexity for the {m.k}-topic model: {m.perplexity:0.4f}")
```

In this instance, the model with more topics has a better perplexity score.
This would suggest that the second model is better fitted to our data and is
thus a "better" model.

But can we do better? What if there's a model with a better score that sits
somewhere between these two topic numbers (or beyond them)? We can test to see
whether this is the case by constructing a `for` loop, in which we iterate
through a range of different topic numbers, train a model on each, and record
the resultant scores.

```{code-cell}
k_range = range(10, 31)
p_scores = []
for k in k_range:
    _model = tp.LDAModel(k = k, corpus = corpus, seed = seed)
    _model.train(iter = iters)
    p_scores.append({'n_topics': k, 'perplexity': _model.perplexity})
```

Convert the results to a DataFrame:

```{code-cell}
p_scores = pd.DataFrame(p_scores)
p_scores.sort_values('perplexity', inplace = True)
p_scores
```

We'll train a new model with the best score.

```{code-cell}
best_k = p_scores.nsmallest(1, 'perplexity')['n_topics'].item()
best_p = tp.LDAModel(k = best_k, corpus = corpus, seed = seed)
best_p.train(iter = iters)
```

Here are the top words:

```{code-cell}
:tags: ["output_scroll"]
for k in range(best_p.k):
    top_words(best_p, k)
```

### Coherence

If you find that your perplexity scores don't translate to interpretable
models, you might use a **coherence score** instead. Coherence scores measure
the degree of semantic similarity among words in a topic. Some people prefer to
use them in place of perplexity because these scores help distinguish between
topics that fit snugly on consistent word co-occurrence and those that are
artifacts of statistic inference.

There are a few ways to calculate coherence. We'll use `c_v` coherence, which
uses the two kinds of text similarity we've already seen: pointwise mutual
information (PMI) and cosine similarity. This method takes the co-occurrence
counts of top words in a given topic and calculates a PMI score for each word.
Then, it looks to every other topic in the model and calculates a PMI score for
the present topic's words and those in the other topics. This results in a
series of PMI vectors, which are then measured with cosine similarity.

Let's look at the score for the best model above:

```{code-cell}
best_p_coherence = Coherence(best_p, coherence = 'c_v')
print(f"Coherence score: {best_p_coherence.get_score():0.4f}")
```

As with perplexity, we can look for the best score among a set of topic
numbers. Here, we're looking for the highest score, which will be a number
between 0 and 1.

```{code-cell}
c_scores = []
for k in k_range:
    _model = tp.LDAModel(k = k, corpus = corpus, seed = seed)
    _model.train(iter = iters)
    coherence = Coherence(_model, coherence = 'c_v')
    c_scores.append({'n_topics': k, 'coherence': coherence.get_score()})
```

Let's format the scores.

```{code-cell}
c_scores = pd.DataFrame(c_scores)
c_scores.sort_values('coherence', ascending = False, inplace = True)
c_scores.head(10)
```

Now we select the best one and train a model on that.

```{code-cell}
best_k = c_scores.nlargest(1, 'coherence')['n_topics'].item()
best_c = tp.LDAModel(k = best_k, corpus = corpus, seed = seed)
best_c.train(iter = iters)
```

Here are the top words for each topic:

```{code-cell}
:tags: ["output_scroll"]
for k in range(best_c.k):
    top_words(best_c, k)
```

And here's a distribution plot of topic proportions. We'll define a function to
create this as we'll be making a few of these plots later on.

```{code-cell}
def format_top_words(tm, k, top_n = 5):
    """Get a formatted string of the top words for a topic."""
    words = tm.get_topic_words(k, top_n = top_n)
    
    return f"Topic #{k}: {', '.join(word for (word, _) in words)}"

def plot_topic_proportions(tm, name = '', top_n = 5):
    """Plot the topic proportions for a model."""
    dists = tm.get_count_by_topics() / tm.num_words
    words = [format_top_words(tm, k, top_n) for k in range(tm.k)]
    data = pd.DataFrame(zip(words, dists), columns = ('word', 'dist'))
    data.sort_values('dist', ascending = False, inplace = True)

    fig, ax = plt.subplots(figsize = (15, 15))
    g = sns.barplot(x = 'dist', y = 'word', color = 'blue', data = data)
    g.set(title = f"Topic proportions for {name}", xlabel = "Proportion");

plot_topic_proportions(best_c, name = 'Best coherence')
```

Fine Tuning: Advanced
---------------------

### Hyperparameters: alpha and eta

The number of topics is not the only value we can set when initializing a
model. LDA modeling has two key **hyperparameters**, which we can configure to
control the nature of the topics a training run produces:

```{margin} Want more details?
This StackExchange [answer][answer] has a remarkably succinct summary of these
two hyperparameters. Additionally, this [blogpost][blogpost] features graphs of
how changes in hyperparameters change the probability distributions in a
Dirichlet space.

[answer]: https://datascience.stackexchange.com/questions/199/what-does-the-alpha-and-beta-hyperparameters-contribute-to-in-latent-dirichlet-a/202#202
[blogpost]: https://blog.bogatron.net/blog/2014/02/02/visualizing-dirichlet-distributions/
```

+ **Alpha**: represents document-topic density. The higher the alpha, the more
  evenly distributed, or "symmetric," topic proportions are in a particular
  document. A lower alpha means topic proportions are "asymmetric," that is, a
  document will have fewer predominating topics, rather than several
+ **Eta**: represents word-topic density. The higher the eta, the more word
  probabilities will be distributed evenly across a topic (specifically, this
  boosts the presence of low-probability words). A lower eta means word
  distributions are more uneven, so each topic will have less dominant words

At core, these two hyperparameters variously control specificity in models: one
for the way models handle document specificity, and one for the way they handle
topic specificity.

```{admonition} On terminology
Different LDA implementations have different names for these hyperparameters.
Eta, for example, is also referred to as beta. When reading the documentation
for an implementation, look for whatever term stands for the "document prior"
(alpha) and the "word prior" (eta).
```

`tomotopy` has actually been setting values for alpha and eta all along. We can
declare them specifically with arguments when initializing a model. Below, we
boost the alpha and lessen the eta. This configuration should give us a more
even distribution in topics among the documents and higher probabilities for
the words in each topic. We'll use the topic number from our best coherence
model above.

```{code-cell}
ae_adjusted = tp.LDAModel(
    k = best_c.k, alpha = 1, eta = 0.001, corpus = corpus, seed = seed
)
ae_adjusted.train(iter = iters)
```

Let's compare with the best coherence model.

```{code-cell}
compare = {'best coherence': best_c, 'high alpha/low eta': ae_adjusted}
for name, tm in compare.items():
    probs = []
    for topic in range(tm.k):
        scores = [s for (w, s) in tm.get_topic_words(topic, top_n = 5)]
        probs.append(scores)

    probs = np.mean(probs)
    words_per_topic = np.median(tm.get_count_by_topics())

    print(f"For the {name} model:")
    print(f"+ Median words/topic: {words_per_topic:0.0f}")
    print(f"+ Mean probablity of a topic's top-five words: {probs:0.4f}%")
```

### Choosing hyperparameter values

In the literature about LDA modeling, researchers have suggested various ways
of setting hyperparameters. For example, the authors of [this paper][finding]
suggest that the ideal alpha and eta values are $\frac{50}{k}$ and 0.1,
respectively (where $k$ is the number of topics). Alternatively, you'll often
see people advocate for an approach called **grid searching**. This involves
selecting a range of different values for the hyperparameters, permuting them,
and building as many different models as it takes to go through all possible
permutations.

[finding]: https://www.pnas.org/doi/10.1073/pnas.0307752101

Both approaches are valid but they don't emphasize an important point about
what our hyperparameters represent. Alpha and eta are _priors_, meaning they
represent certain kinds of knowledge we have about our data before we even
model it. In our case, we're working with book blurbs. The generic conventions
of these texts are fairly constrained, so it probably doesn't make sense to
raise our alpha values. The same might hold for a corpus of tweets collected
around a small keyword set: the data collection is already a form of
hyperparameter optimization. Put another way, _setting hyperparameters depends
on your data and your research question(s)_. It's as valid to ask, "do these
values give me an interpretable model?" as it is to look to perplexity and
coherence scores as the sole arbiters of model quality.

Here's an example of where the interpretability consideration matters. In the
model below, we set hyperparameters to produce low perplexity and coherence
scores.

```{code-cell}
optimized = tp.LDAModel(
    k = best_c.k, alpha = 5, eta = 2, corpus = corpus, seed = seed
)
optimized.train(iter = iters)
optimized_coherence = Coherence(optimized, coherence = 'c_v')
```

The scores look good.

```{code-cell}
print(f"Perplexity: {optimized.perplexity:0.4f}")
print(f"Coherence: {optimized_coherence.get_score():0.4f}")
```

But look at the topics:

```{code-cell}
:tags: ["output_scroll"]
for k in range(optimized.k):
    top_words(optimized, k)
```

And the proportions:

```{code-cell}
plot_topic_proportions(optimized, name = 'Hyperoptimized')
```

The top words are incoherent and one topic all but completely dominates the
topic distribution.

The challenge of setting hyperparameters, then, is that it's a balancing act.
In light of the above output, for example, you might decide to favor
interpretability above everything else. But doing so can lead to overfitting.
Hence the balancing act: the whole process of fine tuning involves
incorporating a number of different considerations (and compromises!) that, at
the end of the day, should work in the service of your research question.

### Final configuration

To return to the question of our own corpus, here are the best topic
number/hyperparameter configuration from a grid search run:

```{code-cell}
grid_df = pd.read_csv(indir.joinpath("grid_search_results.csv"), index_col = 0)
grid_df.sort_values('coherence', ascending = False, inplace = True)
grid_df.head(10)
```

If you look closely at the scores, you'll see that they're all very close to
one another; any one of these options would make for a good model. The
perplexity is a bit high, but the coherence scores look good and the topic
numbers produce a nice spread of topics. Past testing has shown that the
following configuration makes for a particularly good – which is to say,
interpretable – one:

```{code-cell}
tuned = tp.LDAModel(
    k = 29, alpha = 0.1, eta = 0.015, corpus = corpus, seed = seed
)
tuned.train(iter = iters)
tuned_coherence = Coherence(tuned, coherence = 'c_v')
```

Our metrics:

```{code-cell}
print(f"Perplexity: {tuned.perplexity:0.4f}")
print(f"Coherence: {tuned_coherence.get_score():0.4f}")
```

Our topics:

```{code-cell}
:tags: ["output_scroll"]
for k in range(tuned.k):
    top_words(tuned, k)
```

Our proportions:

```{code-cell}
plot_topic_proportions(tuned, name = 'Tuned')
```

Model Exploration
-----------------

Topic proportions and top words are all helpful, but there are other ways to
dig more deeply into a model. This final section will show you a few examples
of this.

First, let's rebuild a theta matrix from the fine-tuned model. Remember that a
theta is a document-topic matrix, where each cell is a probability score for a
document's association with a particular topic.

```{code-cell}
theta = get_theta(tuned, manifest['title'])
```

We'll also make a quick set of labels for our topics, which list the top five
words for each one.

```{code-cell}
labels = [format_top_words(tuned, k) for k in range(tuned.k)]
```

A heatmap is a natural fit for inspecting the probability distributions of
theta.

```{code-cell}
n_samples = 20
blurb_set = manifest['title'].sample(n_samples)
sub_theta = theta[theta.index.isin(blurb_set)]

fig, ax = plt.subplots(figsize = (15, 8))
g = sns.heatmap(sub_theta, linewidths = 1, cmap = 'Blues', ax = ax)
g.set(xlabel = 'Topic', ylabel = 'Title', xticklabels = labels)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
plt.xticks(rotation = 30, ha = 'left')
plt.tight_layout();
```

Topic 15 appears to be about the classics. What kind of titles have a
particularly high association with this topic?

```{code-cell}
def doc_topic_associations(theta, manifest, k):
    """Find highly associated documents from a manifest with a topic."""
    topk = theta.loc[theta.idxmax(axis = 1) == k, k]
    associated = manifest[manifest['title'].isin(topk.index)].copy()
    associated.loc[:, f'{k}_score'] = topk.values

    return associated[['author', 'title', 'genre', f'{k}_score']]

k15 = doc_topic_associations(theta, manifest, 15)
k15.head(15)
```

Certain topics seem to map very nicely onto particular book genres. Here are
some titles associated with topic 13, which is about recipes.

```{code-cell}
k13 = doc_topic_associations(theta, manifest, 13)
k13.head(15)
```

Topic 2 appears to be about girlhood.

```{code-cell}
k2 = doc_topic_associations(theta, manifest, 2)
k2.head(15)
```

Though there are some perplexing titles listed here. What's _The Gospel of the
Flying Spaghetti Monster_ doing here? _The Case Against School Vouchers_ is
similarly odd, though maybe it's a sensible fit given what we might expect from
shared vocabulary. Topic models, remember, are ultimately counting word
co-occurrences, not the different semantic valences of a word, or tone, style,
etc. It's up to us to parse the latter kinds of things.

To do so, it's helpful to examine the overall similarities and differences
between topics, much in the way we projected our documents into a vector space
in the previous chapter. We'll prepare the code to do something similar here
but will save the final result for a separate webpage. Below, we produce the
following:

+ A topic-term distribution matrix (word probabilities for each topic)
+ The lengths of every blurb
+ A list of the corpus vocabulary
+ The corresponding frequency counts for the corpus vocabulary

Once we've made these, we'll prepare our visualization data with a package
called `pyLDAvis` and save it.

```{code-cell}
topic_terms = np.stack([tuned.get_topic_word_dist(k) for k in range(tuned.k)])
doc_lengths = np.array([len(doc.words) for doc in tuned.docs])
vocab = list(tuned.used_vocabs)
term_frequency = tuned.used_vocab_freq

vis = pyLDAvis.prepare(
    topic_terms, theta.values, doc_lengths, vocab, term_frequency
    , start_index = 0, sort_topics = False
)

outdir = indir.joinpath("output/topic_model_plot.html")
pyLDAvis.save_html(vis, outdir.as_posix())
```

With that done, we've finished our initial work with topic models. The
resultant visualization of the above is available [here][vis]. It's a scatter
plot that represents topic similarity; the size of each topic circle
corresponds to that topic's proportion in the model. Explore it some and see
what you find!

[vis]: https://ucdavisdatalab.github.io/workshop_getting_started_with_textual_data/topic_model_plot.html

