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

From Text to Data
=================

This chapter, which we've asked you to read prior to the first workshop
session, is a general discussion of working with textual data in Python. While
the workshop series assumes you have at least a basic understanding of Python,
we'll quickly review how to load, or "read in," a single text file and format
it for text analysis. We'll do so both as a refresher and because this simple
action illuminates an important aspect of working with textual data: namely,
that to your computer, text is above all a **sequence of characters**. This is
a key thing to keep in mind when preparing your data for text mining and/or
NLP.

As you read this chapter, use it as a check on your familiarity with Python. If
you feel comfortable writing the code below, you should be prepared for our
sessions. The skills covered in this chapter include:

+ Loading text data into Python
+ Working with different Python data structures (strings, lists, dictionaries)
+ Control flow with `for` loops

```{tip}
Need to brush up on Python? The DataLab offers a [Python Basics workshop
series][pbws].

[pbws]: https://ucdavisdatalab.github.io/workshop_python_basics
```

Plain Text
----------

To open a text file, we'll use `with...open`.

```{code-cell}
with open("data/session_one/shelley_frankenstein.txt", 'r') as fin:
    frankenstein = fin.read()
```

We use `r` for the `mode` argument because we're working with **plain text**
data (as opposed to **binary** data). In computing, plain text has a fuzzy set
of meanings, but generally it refers to some kind of data that is stored as a
stream of text characters (usually [ASCII][ascii] but increasingly
[UTF-8][utf8]).

[ascii]: https://en.wikipedia.org/wiki/ASCII
[utf8]: https://en.wikipedia.org/wiki/UTF-8

In plain text representations, every keystroke you would use to type out a text
has a corresponding character. That means this print output:

```{code-cell}
print(frankenstein[:364])
```

...is represented by the following plain text:

```{code-cell}
frankenstein[:364]
```

```{margin} You might ask...
Why aren't spaces also represented by characters? Well, they are, but even in
this relatively unformatted view your computer continues to render text in a
human-readable way.
```

See all the `\n`, or **newline**, characters? Each one represents a line break.
On the back end, your computer uses newline characters to demarcate things like
paragraphs, stanzas, titles, and so forth, but typically it suppresses these
characters when it renders text for our eyes. What we see in the two different
outputs above, then, is a difference between **print conventions** and **code
conventions**. What appears as a blank in the former is in fact an addressable
unit in the latter.

Textual Units
-------------

### Characters

The distinction between print and code conventions has significant consequences
for us. While, in the print view of the world, we tend to think of the word as
the atomic unit of text, in the code view of the world, text is – again – a
sequence of characters. The latter view is evident if we count how many units
are in our text:

```{code-cell}
print("The length of Frankenstein is:", len(frankenstein))
```

The Penguin edition of _Frankenstein_ is ~220 pages. If we assume each page has
~350 words, that makes the book ~77,000 words long – far less than the number
above. So why did Python output this number? Because it counted characters, not
words.

### Tokens

But most of the time, we want to work with words. To do so, we'll need to
change how Python represents our data, converting it from a long stream of
characters into discrete units. The process of doing this is called
**tokenization**. _To tokenize_ means to break a continuous sequence of text
data into substrings, or "tokens." Ultimately, tokens are what we primarily
count in text analytics.

Notably, a token is more of a generic entity than it is a particular kind of
text. _Tokens don't always mean words_. In one sense, for example, our text is
already tokenized – it's just tokenized by characters. But we want it tokenized
into words.

There are a number of different Python libraries that can tokenize text for
you, but it's easy enough to do one version of this task with Python alone. For
now, we'll simply use `.split()` and save the result to a new variable, `doc`.
This method breaks text apart on all whitespace characters (`\s`, `\n`, `\t`,
etc.).

```{code-cell}
doc = frankenstein.split()
```

Now, if we call `len()` on `doc`, we'll see this:

```{code-cell}
print("The length of Frankenstein is:", len(doc))
```

Much better! Now that our text is tokenized with whitespaces, this number is
considerably closer to our estimates above.

Counting Words
--------------

With our text data loaded and formatted, it's time to perform one of the core
tasks of text analysis: counting. The next chapter will discuss this process in
greater detail, but we'll preview it here to get a sense of what's to come (and
to review the basics of control flow in Python).

### A first pass

Splitting text transforms it into a list, where each word has its own separate
index position. With this data structure, deriving word counts is as simple as
passing that list to a `Counter`.

```{code-cell}
from collections import Counter

counts = Counter(doc)
```

With this done, we now have the total number of unique words in _Frankenstein_.

```{code-cell}
print("Unique words:", len(counts))
```

We can also access the counts of individual words.

```{code-cell}
to_count = ("imagination", "monster")
for word in to_count:
    print(f"{word:<12} {counts[word]}")
```

```{tip}
For the sake of readability, we use extra string formatting to control the
print spacing of our output. Feel free to work without it. But if you do want
to use it, take a look at [this guide][fguide].

[fguide]: https://realpython.com/python-f-strings/
```

If you're familiar with _Frankenstein_, you'll know that it's an epistolary
novel. That is, it's written as a series of letters. The first word, in fact,
is "Letter." With this in mind, let's tack on "letter" to our loop above.

```{code-cell}
to_count += ("letter",)
for word in to_count:
    print(f"{word:<12} {counts[word]}")
```

### Top words

Everything looks good so far. Let's take a look at the most frequent words in
the novel.

```{code-cell}
:tags: ["output_scroll"]
counts.most_common(50)
```

And there they are!

```{warning}
Except look: do you notice anything strange about these counts? Inspect them
closely. The word "The" appears about 20 words up from the end of the output –
and yet it also appears as the _first_ entry in this output. What's going on?
```

### Investigating duplicates

Let's investigate. To see whether our counts are off, we'll look back at
"letter" from above.

```{code-cell}
print("Count for letter:", counts['letter'])
```

That corresponds to what we saw in the `for` loop. But remember: _Frankenstein_
doesn't start with "letter," it starts with "Letter." Might this make a
difference, as it did with "the" and "The"?

```{code-cell}
print("Count for Letter:", counts['Letter'])
```

We appear to have some duplicates. To diagnose this problem, we'll search
through all unique words in _Frankenstein_ and test whether "letter" is a
substring of another string.

```{code-cell}
for word in counts:
    if "letter" in word:
        print(f"{word:<10} {counts[word]}")
```

For good measure, let's do this with "monster" as well:

```{code-cell}
for word in counts:
    if "monster" in word:
        print(f"{word:<10} {counts[word]}")
```

What, What's a Word?
--------------------

Based on the outputs above, it's clear what has happened. _We have a problem in
the way we've defined the concept of a word_. Remember, to our computers, text
is just a sequence of characters. We had to coax Python intro treating this
sequence as if it were words by splitting it on spaces.

In doing so, we ended up creating a de facto definition of what constitutes a
word: for this definition, a word is any sequence of characters surrounded by
spaces.

If we frame what we've done in this way, we can see that Python followed this
definition perfectly, doing nothing more or less than splitting sequences of
characters on spaces. In a computer's character-by-character way of reading,
"letter" is different from "letter;" – and understandably so, for each is a
different sequence of characters surrounded by spaces. The same goes for
"letter" and "Letter": both are different character sequences surrounded by
spaces, for in a very rudimentary sense, lowercase 'l' and uppercase 'L' are
not the same character. (To be exact, the underlying Unicode
[codepoints][codepoints] for these letters are `U+006C` and `U+004C`,
respectively).

[codepoints]: https://en.wikipedia.org/wiki/Universal_Character_Set_characters

```{margin} To complicate things further...
Non-English languages and non-alphabetic writing systems add a productive
challenge to this topic. We can't cover this in full, but Quinn Drombowski has
written about it in this [blogpost][blog] on text analytics and the "English
default."

[blog]: https://quinndombrowski.com/blog/2020/10/15/whats-word-multilingual-dh-and-english-default/
```

But in another sense, they _are_ the same letter. The problem here arises from
the fact that, as opposed to our computers' highly literal way of reading, we
tend to consider the meaning of words to be something that transcends
differences in capitalization; that is mostly separable from punctuation; and
that sometimes even goes beyond spelling (think American vs. British English)
and inflection ("run," "running," "ran" => "run"). In the output above, what
we'd really like to see is something closer to what linguists call **lexemes**,
or the abstract units of meaning that underlie groups of words. Otherwise,
we're still just counting characters.

The next chapter – and with it, our first workshop session – will discuss how
to prepare textual data so as to begin analyzing words.

