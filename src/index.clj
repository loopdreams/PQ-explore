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

;; The target audience for these kinds of explorations are policymakers (like
;; myself), who are new to RAG/LLMs and want to understand a little more about
;; the details of what goes into this kind of setup.
;;
;; This is mainly intended as an exploratory overview. I used a
;; relatively small range of the potential data (around 10,000 questions
;; spanning less than three months at the beginning of 2024) and I was limited
;; in terms of how much I could explore LLM performance, due to computing
;; constraints (in the case of locally running models) and cost restraints (in
;; the case of cloud-based models).
;;
;; I primarily used the tools provided by the
;; [noj](https://scicloj.github.io/noj/) library, as well as the
;; [langchain4j](https://docs.langchain4j.dev/) library in the case of vector
;; embeddings. I also received huge support and guidance from the
;; [scicloj](https://scicloj.github.io/) community, which I am deeply grateful
;; for.
