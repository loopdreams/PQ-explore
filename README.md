# PQ Data Explorations

!Work-in-progress

## Goal 

As part of the Irish parliamentary system, members of parliament are able to submit written or spoken questions to Ministers and their corresponding departments. These questions and answers are part of the public record. 

At the practical level, preparing answers to questions can often involve a degree of duplication. The initial goal of this project was to **develop an approach for quantifying levels of duplication across the records**. It should be noted that, in this context, duplication should be taken as a neutral term. In most cases some degree of duplication is necessary to ensure consistency and improve efficiencies at the department side. 

At the same time, the actual process of ensuring this consistency at the administrative level (searching through and compositing previous responses and public information), can be time consuming. For this reason, the secondary goal of this project is to also **develop a RAG framework for an LLM** which could automate part of this common administrative task. 

In this context, the approaches to detecting duplication can be re-used to check LLM responses to **ensure consistency** with previous answers. In this case, high levels of 'duplication' are positive if the question asked is very similar to previously asked questions.

The actual implementation of this could involve:
- LLM generates response to question, using RAG architecture
- The 'validation' step is applied to the answer so that, for example, parts of the answer that appear 'dissimilar' to existing corpus can be visually flagged for human review

The existing questions/answers that are available contain additional metadata that can be helpful in building a RAG model, including:
- Date question was asked 
- General Topic of the question
- Who asked the question
- Which department/minister answered the question



## Build/Render the Notebook
### Install Python Dependencies

This notebooks depends on a few python functions. Full instructions for using python with clojure are available [at the libpython-clj respository.](https://github.com/clj-python/libpython-clj)

Below, I'll go over the steps that I took (I use MacOS and Emacs)

1. (Optional) Set up a python virtual environment 

In my case, I used [pyenv](https://github.com/pyenv/pyenv). 

``` sh
brew install pyenv pyenv-virtualenv
```

Then, create a virtual environment (I used python 3.12.1).

``` sh
pyenv virtualenv 3.12.1 venv-name
```

Activate it with:

``` sh
pyenv activate venv-name
```

2. Install Dependencies 

This project depends on:
- [nltk](https://www.nltk.org/)
- [continuous-eval](https://github.com/relari-ai/continuous-eval/tree/main?tab=readme-ov-file)

``` sh
python3 -m pip install continuous-eval nltk
```

3. Load these in clojure using libpython-clj 

In my case I did this by adding the following to a `dev/user.clj` file. Replace the path references to path to the relevant python binary and library folder (where you installed the dependencies above)

``` clojure
(ns user
  (:require [libpython-clj2.python :as py]))



(py/initialize! :python-executable (str (System/getenv "HOME") "/.pyenv/versions/3.12.1/envs/VENV-NAME/bin/python3.12")
                :library-path (str (System/getenv "HOME") "/.pyenv/versions/3.12.1/lib/python3.12/site-packages/"))
```

### Build the notebook 

Run the following command, which will create the notebook in a `book` directory and start a server with clay.

``` clojure
clj -X:make-book
```

