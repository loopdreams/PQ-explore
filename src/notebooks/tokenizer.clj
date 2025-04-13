(ns notebooks.tokenizer
  (:require [clojure.set :as set]
            [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python :refer [py. py.. py.-] :as py]
            [clojure.string :as str])
  (:import (dev.langchain4j.data.embedding Embedding)
   (dev.langchain4j.data.segment TextSegment)
   (dev.langchain4j.model.embedding EmbeddingModel)
   (dev.langchain4j.store.embedding EmbeddingMatch)
   (dev.langchain4j.store.embedding.inmemory InMemoryEmbeddingStore)
   (dev.langchain4j.model.embedding.onnx.allminilml6v2 AllMiniLmL6V2EmbeddingModel)))

(require-python '[nltk.tokenize.destructive :refer [MacIntyreContractions]]
                '[re :refer [compile]])


;; Taken from python/string punctuation
(def punctuation "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

(def stopwords-english (str/split-lines (slurp "data/nltk-stopwords.txt")))

(defn remove-punctuation [text]
  (let [marks (mapv identity punctuation)]
    (->> text
         (filter (fn [char] (not (some #{char} marks))))
         (str/join ""))))

(defn remove-stopwords [text]
  (->> (str/split text #" ")
       (filter (fn [word] (not (some #{word} stopwords-english))))
       (str/join " ")))

(defn handle-contractions [text]
  (let [contractions (MacIntyreContractions)
        cont-2 (->> (py.- contractions CONTRACTIONS2)
                    (mapv compile))
        cont-3 (->> (py.- contractions CONTRACTIONS3)
                    (mapv compile))
        cont-all (concat cont-2 cont-3)]
    (reduce (fn [txt contraction]
              (py. contraction sub " \\1 \\2 " txt))
            text cont-all)))

(def prepare-text
  (comp remove-punctuation remove-stopwords handle-contractions str/lower-case))

(def is-number-re (re-pattern  "^[+-]?(\\d+(\\.\\d*)?|\\.\\d+)([eE][+-]?\\d+)?$"))

(defn simple-tokenizer-words [text]
  (let [text (prepare-text text)]
    (->>
     (for [word (str/split text #" ")]
       (if (re-find is-number-re word)
         (float (parse-double word))
         word))
     (remove #(= % "")))))

;; Taken from stackoverflow (and updated) - https://stackoverflow.com/questions/25735644/python-regex-for-splitting-text-into-sentences-sentence-tokenizing
(def sentence-splitter-re (re-pattern "(?<!\\w\\.\\w.)(?<!\\b[A-Z][a-z]\\.)(?<![A-Z]\\.)(?<=\\.|\\?|\\!)\\s|\\\n"))

(defn simple-tokenizer-sentences [text]
  (mapv prepare-text (str/split text sentence-splitter-re)))

(comment
  (simple-tokenizer-sentences "Hello World! Goodbye mars. Something else")
  (simple-tokenizer-words "Hello world!"))


;; Metrics

(defn recall [t-s1 t-s2]
  (if (and (seq t-s1) (seq t-s2))
    (->
     (/ (count (set/intersection (set t-s1) (set t-s2)))
        (count (set t-s1)))
     float)
    0))

(defn precision [t-s1 t-s2]
  (if (and (seq t-s1) (seq t-s2))
    (->
     (/ (count (set/intersection (set t-s1) (set t-s2)))
        (count (set t-s2)))
     float)
    0))

(defn IoU [t-s1 t-s2]
  (if (and (seq t-s1) (seq t-s2))
    (let [overlap (count (set/intersection (set t-s1) (set t-s2)))]
      (->
       (/ overlap
          (- (+ (count t-s1) (count t-s2))
             overlap))
       float))
    0))

(defn calculate-retrieval-metrics [target retrieved kind & label]
  (if (not (some #{kind} [:word :sentence]))
    "Error: incorrect kind selected, should be :word or :sentence."
    (let [tokenizer-fn (if (= kind :word) simple-tokenizer-words simple-tokenizer-sentences)
          [hl-token rt-token] (mapv tokenizer-fn [target retrieved])]
      {:recall (recall hl-token rt-token)
       :precision (precision hl-token rt-token)
       :IoU (IoU hl-token rt-token)
       :label (first label)})))


(comment
  (let [target "That winter Robert Cohn went over to America with his novel"
        retrieved "That winter Robert Cohn went over to America with his novel, and it was accepted by a fairly good publisher."]
    (calculate-retrieval-metrics target retrieved :word))
  (let [target "That winter Robert Cohn went over to America with his novel, and it was accepted by a fairly good publisher. His going made an awful row I heard, and I think that was where Frances lost him, because several women were nice to him in New York, and when he came back he was quite changed. He was more enthusiastic about America than ever, and he was not so simple, and he was not so nice. The publishers had praised his novel pretty highly and it rather went to his head. Then several women had put themselves out to be nice to him, and his horizons had all shifted. For four years his horizon had been absolutely limited to his wife. For three years, or almost three years, he had never seen beyond Frances. I am sure he had never been in love in his life. "
        retrieved "
That winter Robert Cohn went over to America with his novel, and it was accepted by a fairly good publisher. His going made an awful row I heard, and I think that was where Frances lost him, because several women were nice to him in New York, and when he came back he was quite changed. He was more enthusiastic about America than ever, and he was not so simple, and he was not so nice. The publishers had praised his novel pretty highly and it rather went to his head. Then several women had put themselves out to be nice to him, and his horizons had all shifted. For four years his horizon had been absolutely limited to his wife. For three years, or almost three years, he had never seen beyond Frances. I am sure he had never been in love in his life.

He had married on the rebound from the rotten time he had in college, and Frances took him on the rebound from his discovery that he had not been everything to his first wife. He was not in love yet but he realized that he was an attractive quantity to women, and that the fact of a woman caring for him and wanting to live with him was not simply a divine miracle. This changed him so that he was not so pleasant to have around. Also, playing for higher stakes than he could afford in some rather steep bridge games with his New York connections, he had held cards and won several hundred dollars. It made him rather vain of his bridge game, and he talked several times of how a man could always make a living at bridge if he were ever forced to. "]
    (calculate-retrieval-metrics target retrieved :sentence)))
