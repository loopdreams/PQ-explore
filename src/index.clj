(ns index)

;; ![Dáil Éireann (The Irish Parliament)](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/DailChamber_2020.jpg/2560px-DailChamber_2020.jpg)
;;
;; # Introduction
;;
;; As part of its democratic functions, the Irish Parliament (Dáil Éireann) has a
;; process for members to ask questions of Ministers and their departments.
;;
;; Ministers must provide answers promptly, and these questions and answers
;; are subsequently part of the public record. To see examples of these questions
;; and answers, see the [Oireachtais website](https://www.oireachtas.ie/en/debates/questions/).
;;
;; The goal of this project was to explore these questions and answers using
;; some standard Retrieval Augmented Generation (RAG) techniques.
;;
;; Firstly, I looked at how to store questions in a **vector database**. I also
;; explored some visualising techniques to try to help build intuition about
;; what is happening when questions are transformed into vector embeddings.
;;
;; Next, I simulated a standard RAG setup by using this question database to
;; provide a LLM with context for generating its own response. I then explored
;; various rudimentary **validation** techniques to try to see how the models
;; perform with this kind of task. 

;; This is mainly intended as an exploratory overview. I used a
;; relatively small range of the potential data (around 10,000 questions
;; spanning less than three months at the beginning of 2024). 
;;
;; I primarily used the tools provided by the
;; [noj](https://scicloj.github.io/noj/) library, as well as the
;; [langchain4j](https://docs.langchain4j.dev/) library in the case of vector
;; embeddings. I also received huge support and guidance from the
;; [scicloj](https://scicloj.github.io/) community, which I am deeply grateful
;; for.

;; ## Key Questions
;;
;; ### Could an LLM help optimize a common administrative task?
;;
;; Answering questions from members of the public about work your organisation
;; does can often be a time-consuming administrative task. A certain amount of
;; questions involve re-using or re-compiling material from previously-answered
;; questions that are similar.
;;
;; LLMs are great at quickly summarising information, so the basic question here
;; is wheather an LLM could potentially help optimize this daily task.
;;
;; Using the RAG approach helps us restrict the answers to only information that
;; has already been previously provided.
;;
;; ### Can we do this with `clojure`?
;;
;; This might seem like an unimportant question (of course you can do it in
;; clojure!). However, because the current ecosystem around LLM-based
;; datascience is heavily weighted toward `python`, it is important to
;; acknowledge here that a lot of the time for this project was spent on
;; 'translating' some of these approaches that have been developed in python
;; into clojure.
;;
;; ### How reliable is it?
;;
;; RAG-based applications are relatively straight-forward to set up. However, the
;; main questions usually center on how accurate and reliable the answers are.
;; This is especially true when it comes to public policy.
;;
;; There are already lots of approaches for testing this kind of setup,
;; including frameworks like [RAGAS.](https://docs.ragas.io/en/stable/) Here, I
;; try out a few evaluation techniques.
;;
;; The analysis here is certainly not exhaustive. Hopefully it can help point in
;; the direction of more complete evaluation strategies using clojure tools.
;;
;;
;;
;; ## Notebooks Overview

;; **1. Dataset Preparation**
;; 
;; Some basic cleaning/preparation steps on the source dataset.
;;
;; **2. Vector Database**
;; 
;; Creating a vector database based on the 'questions' column of the dataset.
;;
;; **3. Retrieval Evaluation**
;;
;; Applying some evaluation metrics to our vector database. Trying out different
;; retrieval strategies.
;;
;; **4. Answer Generation**
;;
;; Using the vector database to provide context to an LLM and testing answer generation.

;; **5. Generation Evaluation**
;;
;; Examining what kinds of metrics we can use to evaluate the LLM-generated answers.
;;
;; **6. Single Model Evaluation**
;;
;; Putting together the whole process and evaluating some models/generation strategies.
