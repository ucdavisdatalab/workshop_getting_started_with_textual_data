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

Before We Begin...
==================

This reader is meant to serve both as a roadmap for the overall trajectory of
the series and as a reference for later work you may do in text mining. Our
sessions will follow its overall logic, but the reader itself offers
substantially more details than we may have time to discuss in the sessions.
The instructors will call attention to this when appropriate; you are
encouraged to consult the reader whenever you would like to learn more about a
particular topic.

Each session of this workshop will cover material from one or more chapters. We
also ask that you **read Chapter 2 in advance of our first session**. It's a
review of sorts and will set up a frame for the series.

| Session | Date | Chapters Covered | Topic                                    |
| ------- | ---- | ---------------- | ---------------------------------------- |
|    0*   |  --  |     Chapter 2    | Review: textual data in Python           |
|    1    | 2/18 |     Chapter 3    | Text cleaning                            |
|    2    | 2/20 |  Chapters 4 & 5  | Corpus analytics and document clustering |
|    3    | 2/22 |     Chapter 6    | Topic modeling                           |

\* Please read in advance

```{admonition} Learning Objectives
By the end of this series, you will be able to:

+ Prepare textual data for analysis using a variety of cleaning processes
+ Recognize and explain how these cleaning processes impact research findings
+ Explain key terminology in text mining, including "tokenization," "n-grams,"
  "lexical diversity," and more
+ Use special data structures (such as document-term matrices) to efficiently
  analyze multiple texts
+ Use statistical measures (pointwise mutual information, tf-idf) to identify
  significant patterns in text
+ Cluster and classify texts on the basis of such measures
+ Produce statistical models of "toics" from/about a collection of texts
```

File and Data Setup
-------------------

### Google Colab

We will be using Google Colab's platform and Google Drive during the series and
working with a set of pre-configured notebooks and data. You must have a Google
account to work in the Colab environment. Perform the following steps to setup
your environment for the course:

1. Download the [data][zipped]
2. Un-compress the downloaded .zip file by double clicking on it
3. Visit the Google Colab [website][site] at and sign-in using your Google
   account
4. In a separate browser tab (from the one where you are logged-in to Colab)
   sign-in to your Google Drive
5. Upload the `tm_workshop_data` directory into the root of your Google Drive

[zipped]: https://datalab.ucdavis.edu/tm_workshop_data.zip
[site]: https://colab.research.google.com

Once you have completed the above steps, you will have your basic environment
setup. Next, you'll need to create a blank notebook in Google Colab. To do
this, go to Google Colab and choose "File->New Notebook" from the File Menu.
Alternatively, select "New Notebook" in the bottom right corner of the
notebooks pop-up if it appears in your window.

Now, you need to connect your Google Drive to your Colab environment. To do
this, run the following code in the code cell at appears at the top of your
blank notebook:

```
from google.colab import drive
drive.mount('/gdrive')
```

Your environment should be ready to go!

### Template code

This workshop is hands-on, and you're encouraged to code alongside the
instructors. That said, we'll also start each session with some template code
from the session before. You can find these templates in this [start
script][ss] directory. Simply copy/paste the code from the `.txt` files into
your Jupyter environment.

[ss]: https://github.com/ucdavisdatalab/workshop_getting_started_with_textual_data/tree/main/start_scripts

Assessment
---------------

If you are taking this workshop to complete a GradPathways [micro-credential
track][microcredential], you can find instructions for the assessment
[here][here].

[microcredential]:https://gradpathways.ucdavis.edu/micro-credentials
[here]: https://github.com/ucdavisdatalab/workshop_getting_started_with_textual_data/tree/main/assessment

