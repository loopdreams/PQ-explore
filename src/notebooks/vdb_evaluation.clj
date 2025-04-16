(ns notebooks.vdb-evaluation
  (:require [clojure.edn :as edn]
            [notebooks.preparation :refer [ds]]
            [notebooks.question-vdb :refer [add-question-to-store! db-store query-db-store]]
            [scicloj.kindly.v4.kind :as kind]
            [notebooks.tokenizer :as tokenizer]
            [scicloj.tableplot.v1.plotly :as plotly]
            [clojure.string :as str]
            [tablecloth.api :as tc])
  (:import
   (dev.langchain4j.data.segment TextSegment)
   (dev.langchain4j.store.embedding.inmemory InMemoryEmbeddingStore)
   (dev.langchain4j.model.embedding.onnx.allminilml6v2 AllMiniLmL6V2EmbeddingModel)))


(comment
  ;; Interesting how many matches there are for this! Shows the duplication of language
  (filter #(re-find (re-pattern (str/join "|" highlights)) (second %))
          (map-indexed vector (remove nil? (:answer ds)))))

;; ## Sample Answers and Questions
;;
;; For this exercise, we'll first create a small sample of answers
;; ('highlights') and questions that we will use to test the performance of the
;; retrieval method.


(def highlights-answers
  ["Under the 2019 GP Agreement additional annual expenditure provided for general practice has been increased now by €211.6m."
   "As per HSE eligibility criteria, the educational requirement for a Health Care Assistant is the relevant Health Skills Level 5 (QQI) qualification."
   "The national drug strategy, Reducing Harm, Supporting Recovery, sets out government policy on drug and alcohol use for the period 2017 to 2025."
   "However, local authorities were invited to submit up to 5 applications to the value of €1.5 million per local authority."
   "The salary scale for an archaeologist in the local government sector ranges from €55,519 to €77,176. "])

;; Original Questions
;; - "Deputy Paul Kehoe asked the Minister for Health to address the provision of GP services in the Navan Road, Pelletstown and Ashtown areas of Dublin West, where people have reported difficulty in accessing GP services and complain that the list of doctors they have been provided with is out of date, including retired and in some cases deceased GPs, and where a medical centre lies empty in Pelletstown; and if he will make a statement on the matter."
;; - "Deputy Patricia Ryan asked the Minister for Health if his Department plans to put in place core funding for the provision of Level 6 (QQI) courses for healthcare assistants; and if he will make a statement on the matter."
;; - "Deputy Thomas Gould asked the Minister for Rural and Community Development the role her Department is taking in the National Drugs Strategy."
;; - "Deputy Donnchadh Ó Laoghaire asked the Minister for Rural and Community Development how her Department will work with local authorities to encourage them to apply for new funding for the town and village renewal scheme; if there will be a minimum submission requirement for each local authority; and if she will make a statement on the matter."
;; - "Deputy Aengus Ó Snodaigh asked the Minister for Housing, Local Government and Heritage the estimated cost of hiring a qualified archaeologist for every local authority."

(def highlights-questions
  ["What is the government doing to help improve GP services?"
   "Will the government put in place Level 6 (QQI) courses for healthcare assistants?"
   "What is the government doing with regard to the National Drugs Strategy?"
   "How is the government encouraging local authorities to apple for the town and village renewal scheme?"
   "What is the salary scale for an archaeologiest in the local government sector?"])

;; At the moment, our approach is based on *searching for similar questions*,
;; and then returning their answers.  However, this is a bit of a 'naive'
;; approach. It is based on the concrete steps that are usually taken when trying to answer a new question:
;;
;; 1. Search for previous similar questions
;;
;; 2. Scan these answers for relevant info
;;
;; Why not just skip this step of searching through questions all together? The
;; approach chosen here might depend on the actual application of a system like
;; this. For example, would the intended use be to help administors prepare
;; answers to new questions? In this case searching through previous questions
;; might be more useful. If the intended use is something closer to 'general'
;; retrieval of information, then the best approach might be to simply search
;; through all previous answers for the info.
;;
;; In order to determine the most optimal approach, there are some metrics we
;; could introduce to test the performance of different methods.
;;
;; As a starting point, let's see what kind of information the system we've
;; built so far returns. We'll compare these responses to a more optimized
;; approach at the end.

(->
 (query-db-store "What is the government doing to help improve GP services?" 5)
 (tc/dataset)
 (tc/map-columns :answer [:text] (fn [t] (-> ds
                                             (tc/select-rows #(= t (:question %)))
                                             :answer
                                             first)))
 (kind/table))

;; As we can see, we do get some relevant information, but we also get some
;; unhelpful information (like the first answer).
;;
;; Also, depending on how specific our question is, this is potentially
;; way to much information, and could potentially confuse or mislead the LLM
;; later on.
;;
;; To help improve this, let's first take a look at some common, simple metrics
;; that are used to measure retrieval:

;; [Description of the metrics]


;; Simply splits into sentences and group sentences by 'chunk size'
(defn chunking-fn [documents chunk-size]
  (let [sentences (->> (str/split documents tokenizer/sentence-splitter-re)
                       (remove #(= (str/trim %) "")))
        chunks (->> (partition-all chunk-size sentences)
                    (mapv #(str/join #" " %)))]
    chunks))


(def embedding-model (AllMiniLmL6V2EmbeddingModel/new))

(defn chunked-docs [docs chunking-size]
  (->> (mapv #(chunking-fn % chunking-size) docs) ;; In this case chunk the documents individually, because we know that are all separate/discrete answers
       (remove empty?)
       (reduce into)))


;; TODO: add ability to use openai embeddings and test with openai embeddings

(defn generate-metrics [highlights chunked-docs & label]
  (let [db-store     (InMemoryEmbeddingStore/new)
        num          (count (map #(add-question-to-store! % db-store) chunked-docs))
        _            (println num)]
    (loop [[h & hs] highlights
           results  []]
      (if-not h
        results
        (let [hl-embedding (->> (TextSegment/from h)
                                (. embedding-model embed)
                                (.content))
              matches      (->> (. db-store findRelevant hl-embedding 5)
                                ;; (filterv #(> (.score %) 0.7))
                                (mapv #(.text (.embedded %)))
                                (str/join " "))]
          (recur hs
                 (conj results (tokenizer/calculate-retrieval-metrics h matches :word (first label)))))))))

(defn generate-metrics-question-retrieval-method [highlights hl-questions]
  (loop [idx 0
         res []]
    (if (= idx (count highlights))
      res
      (let [q-embedding       (->> (TextSegment/from (nth hl-questions idx))
                                   (. embedding-model embed)
                                   (.content))
            q-matches         (->> (. db-store findRelevant q-embedding 5)
                                   (mapv #(.text (.embedded %))))
            corresponding-ans (->> (tc/select-rows ds #(some #{(:question %)} q-matches))
                                  :answer
                                  (str/join " "))]
        (recur (inc idx) (conj res (tokenizer/calculate-retrieval-metrics
                                    (nth highlights idx)
                                    corresponding-ans
                                    :word
                                    "Question Method")))))))


(comment
  (def met-comparisons
    (let [hls highlights-answers
          docs (-> ds
                   (tc/drop-missing :answer)
                   (tc/drop-rows #(re-find #"details supplied" (% :question)))
                   (tc/drop-rows #(re-find #"As this is a service matter" (% :answer)))
                   :answer)
          full-docs-benchmark (generate-metrics hls docs "Full Docs")]
      (loop [[x & xs] [3 5 10 15]
             result []]
        (if-not x
          (conj result full-docs-benchmark)
          (let [chunked-docs (chunked-docs docs x)]
            (recur xs
                 (conj result (generate-metrics hls chunked-docs (str x " Chunks")))))))))

  (spit "data/metrics_retrieval_data.edn" (into (reduce into met-comparisons)
                                                (generate-metrics-question-retrieval-method highlights-answers highlights-questions))))

(def comparison-data (edn/read-string (slurp "data/metrics_retrieval_data.edn")))

(kind/table comparison-data)

(defn average [coll]
  (float
   (/ (apply + coll)
      (count coll))))



(def temp-ds
  (->
   (tc/dataset comparison-data)
   (tc/group-by [:label])
   (tc/aggregate {:avg-recall #(average (% :recall))
                  :avg-precision #(average (% :precision))
                  :avg-IoU #(average (% :IoU))})))

(-> temp-ds
    (plotly/layer-line
     {:=x :label
      :=y :avg-recall}))

(-> temp-ds
    (plotly/layer-line
     {:=x :label
      :=y :avg-precision}))

(-> temp-ds
    (plotly/layer-line
     {:=x :label
      :=y :avg-IoU}))

;; In the graphs above, the results for 'recall' are very surprising. Generally,
;; you would expect recall to go up with longer chunks of text (i.e., there
;; would be more likihood of capturing the highlighted text). A major caveat
;; here is that these would need to be run on a much larger test dataset to get
;; something a bit more tangible.
;;
;; For now, let's just be happy that these results point to a clear optimum
;; retrieval strategy for this dataset - splitting the answers into 3-sentence
;; chuncks results in both high recall and relatively high precision.

;; Based on the above, let's try to generate context using a chunking method
;; that divides the answer text into chunks of three sentences.

(comment
  (let [answers (-> ds
                    (tc/drop-missing :answer)
                    (tc/drop-rows #(re-find #"details supplied" (% :question)))
                    (tc/drop-rows #(re-find #"As this is a service matter" (% :answer)))
                    :answer)

        docs (chunked-docs answers 3)
        db-store (InMemoryEmbeddingStore/new)
        _c (count (mapv #(add-question-to-store! % db-store) docs))]
    (println _c)
    (spit "data/retrieval_store/store.json" (.serializeToJson db-store))))

(def db-store-chunked-answers (InMemoryEmbeddingStore/fromFile "data/retrieval_store/store.json"))


(defn generate-context [question]
  (let [emb-question (.content (. embedding-model embed question))
        related-docs (. db-store-chunked-answers findRelevant emb-question 5)]
    (map (fn [doc]
           {:text (.text (.embedded doc))
            :score (.score doc)})
         related-docs)))

(kind/table
 (generate-context "What is the government doing to help improve GP services?"))

;; These answers are not a bad starting point for answering this kind of broad
;; question. You can see some duplication in the answers, which, if this were to
;; be optimized further should be removed from the database to improve results further.
;; 
;; Looking at the first answer in the table above, the figure of '211m EUR' is
;; referenced in relation to the 2019 GP Agreement. Let's see if the database
;; can match this exact figure:

(kind/table
 (generate-context "How much annual investment was provided under the 2019 GP agreement?"))

;; Every document retrieved seems to contain the relevant figure :)
;;
;; For completness, let's try this same, more specific, question with the
;; previous approach. As you can see below It's much less focused! 


(->
 (query-db-store "How much annual investment was provided under the 2019 GP agreement" 5)
 (tc/dataset)
 (tc/map-columns :answer [:text] (fn [t] (-> ds
                                             (tc/select-rows #(= t (:question %)))
                                             :answer
                                             first)))
 (kind/table))
