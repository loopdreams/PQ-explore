(ns notebooks.word2vec
  (:require [clojure.string :as str]
            [tablecloth.api :as tc]
            [libpython-clj2.python :as py :refer [py.- py. py..]]
            [libpython-clj2.require :refer [require-python]])
  (:import [org.deeplearning4j.text.sentenceiterator BasicLineIterator]
           [org.deeplearning4j.models.word2vec Word2Vec$Builder]
           [org.deeplearning4j.models.paragraphvectors ParagraphVectors$Builder]
           [org.deeplearning4j.text.tokenization.tokenizer.preprocessor CommonPreprocessor]
           [org.deeplearning4j.text.tokenization.tokenizerfactory DefaultTokenizerFactory]
           [org.deeplearning4j.models.embeddings.loader WordVectorSerializer]
           [org.nd4j.linalg.factory Nd4j]
           [org.datavec.api.util ClassPathResource]))

;; Construct Text File for Model

;; Around 10K answers
(comment
  (->>
   (-> (tc/dataset "data/20250302_PQs_10K_2024_answers.csv" {:key-fn keyword})
       (tc/drop-missing :answer))
   :answer
   (str/join "\n")
   (spit "data/raw_answers.txt")))

;; Additional cleaning on answers needed:
;; - Remove Irish language questions
;; - Remove 'composite' qustions
;; - Remove questions that are 'redirecting' to HSE
;; - Remove tags where missing xml info




;; Build word2vec model - a basic version following this blog post:
;; http://www.appliedscience.studio/articles/dl4j-word2vec-clj.html

(def model
  (-> (Word2Vec$Builder.)
      (.minWordFrequency 5)
      (.iterations 1)
      (.layerSize 100)
      (.seed 42)
      (.windowSize 5)
      (.iterate (BasicLineIterator. "data/raw_answers.txt"))
      (.tokenizerFactory (doto (DefaultTokenizerFactory.)
                           (.setTokenPreProcessor (CommonPreprocessor.))))
      (.build)))

(.fit model)

(comment
  ;; Not sure how to properly use the paragraph parts of this library yet...
  (def model-para
    (-> (ParagraphVectors$Builder.)
        (.minWordFrequency 1)
        (.layerSize 100)
        (.seed 42)
        (.iterate (BasicLineIterator. "data/raw_answers.txt"))
        (.tokenizerFactory (doto (DefaultTokenizerFactory.)
                             (.setTokenPreProcessor (CommonPreprocessor.))))
        (.build))))



(comment
  (.fit model-para))



;; Associated Words
(.wordsNearest model "children" 10)

(.wordsNearest model "funding" 10)

(.wordsNearest model "environmental" 10)

(.wordsNearest model "department" 10)

;; Similarities?
(.similarity model "children" "child")

(.similarity model "land" "property")

(.similarity model "support" "assistance")

(.similarity model "EU" "European Union")

(.similarity model "scheme" "programme")

;; A good sign with this one, properly recognises higher similarity between these two words
;; which both mean the same thing in this context.
(.similarity model "minister" "td")

(.similarity model "funding" "budget")



;; Next - look into deeplearning4j more and use the doc2vec functionalities to build a model
;; that can compare similarities between sentences.
;;
;; Purpose - in a RAG/LLM generated response, search through a pre-defined 'sentence' database to ensure that
;; semantically similar sentences can be found (and list these for human review).
;;
;; Eventually, it would be better if entire answer/doc could be checked, but let's try sentence by sentence first.
;;
;; For example (a real phrase from the answers text):
;;"We look forward to playing a full and active role in the 2026 EU Presidency."
;;
;; A. We do not wish to play a full and active role in the 2026 EU Presidency. -> should be highly dissimilar to sentences in the database (even though contains many similar phrases)
;;
;; B. The Government is looking forward to playing an active role in the 2026 EU Presidency -> should be highly similar
;;

;; Trying with python (Gensim)


(defn answers->sentences [answers]
  (reduce (fn [res answer]
            (into res (str/split (str answer) #"\. (?=[A-Z])")))
          []
          answers))


(def PQ-sentences
  (-> (tc/dataset "data/20250302_PQs_10K_2024_answers.csv" {:key-fn keyword})
      (tc/drop-missing :answer)
      :answer
      distinct
      answers->sentences))

(def PQ-full-docs
  (-> (tc/dataset "data/20250302_PQs_10K_2024_answers.csv" {:key-fn keyword})
      (tc/drop-missing :answer)
      :answer
      distinct))

(require-python '[gensim.models.doc2vec :refer [Doc2Vec TaggedDocument]]
                '[nltk.tokenize :refer [word_tokenize]])


;; Sometimes causes repl to crash:
(comment
  (require-python '[nltk :as nltk])
  (nltk/download "punkt")
  (nltk/download "punkt_tab"))


(def PQ-sentences-tokenized
  (->> PQ-sentences
       (mapv str/lower-case)
       (mapv word_tokenize)))

(def PQ-sentences-tagged
  (let [indexed (map-indexed vector PQ-sentences-tokenized)]
    (mapv (fn [[idx words]] (TaggedDocument :words words :tags (str idx))) indexed)))


;; Low num of epochs here while just getting things set up, because it takes a long time to run otherwise
(def model (Doc2Vec :vector_size 100 :window 2 :min_count 1 :workers 4 :epochs 5))

(comment
  (py. model build_vocab (first PQ-sentences-tagged))
  (py. model build_vocab  PQ-sentences-tagged))


;; Not sure why, but evaling this sometimes causes repl to crash
(comment
  (py. model train PQ-sentences-tagged
       :total_examples (py.- model corpus_count)
       :epochs (py.- model epochs)))

;; Major issue with the examples below - they all seem to return documents at the top of the list (eg. between indexes 0-10)

;; Similar to one of the sentences in db
(def new-sentence-similar "I will to work with members of the house in progressing legislation")
;; Similar to sentence, but with a negative sentiment
(def new-sentence-negative "I will not work with other people when it comes to passing laws")
;; A nonsense sentence (Lewis Carrol)
(def new-sentence-dissimilar "Just the place for a Snark! the Bellman cried, As he landed his crew with care")

(defn print-sentence-comparisons-repl [sentence]
  (let [inferred-vec (->> (str/lower-case sentence)
                          (word_tokenize)
                          (py. model infer_vector))
        similar-docs (py. (py.- model dv) most_similar inferred-vec :topn 5)]
    (println "-------------------------------------------------")
    (println (str "Candidate: " sentence))
    (doseq [[idx score] similar-docs]
      (println (str "Document " idx ": Similarity Score: " score))
      (println (str "Text:" (nth PQ-sentences (parse-long idx)))))))

(comment
  (print-sentence-comparisons-repl new-sentence-similar)
  (print-sentence-comparisons-repl new-sentence-negative)

  (print-sentence-comparisons-repl new-sentence-dissimilar))


(comment
  (def test-sentence-drs "We are encouraging more people to rcycle plastic bottles to help meet EU targets.")

  (print-sentence-comparisons-repl test-sentence-drs)

  (def inferred-vec-2 (->> (str/lower-case test-sentence-drs)
                           (word_tokenize)
                           (py. model infer_vector)))
  (def similar-drs (py. (py.- model dv) most_similar inferred-vec-2 :topn 20))
  (nth PQ-sentences 3))
