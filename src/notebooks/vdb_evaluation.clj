(ns notebooks.vdb-evaluation
  (:require [clojure.set :as set]
            [notebooks.preparation :refer [ds]]
            [notebooks.question-vdb :refer [add-question-to-store! db-store]]
            [notebooks.tokenizer :as tokenizer]
            [scicloj.tableplot.v1.plotly :as plotly]
            [clojure.string :as str]
            [tablecloth.api :as tc])
  (:import (dev.langchain4j.data.embedding Embedding)
   (dev.langchain4j.data.segment TextSegment)
   (dev.langchain4j.model.embedding EmbeddingModel)
   (dev.langchain4j.store.embedding EmbeddingMatch)
   (dev.langchain4j.store.embedding.inmemory InMemoryEmbeddingStore)
   (dev.langchain4j.model.embedding.onnx.allminilml6v2 AllMiniLmL6V2EmbeddingModel)))

;; ## Types of Metrics
;; TODO: overview



;; ## Chroma Evaulation

;; Install the chroma repo:
;; - !pip install git+https://github.com/brandonstarxel/chunking_evaluation.git



;; Some factual data that is found within the store of answers
(def highlights
  ["On the 26 April 2023 the EU Commission published its proposal to revise the general pharmaceutical legislation"
   "The ARP is paid at the rate of €800 per month per property with a unique Eircode."
   "Between 01 January 2023 and 31 December 2023, 100% of the videos posted on the Department's social media channels included subtitles."
   "Expressions of interests were received from over 900 primary schools in respect of 150,000 children and late last year these schools were invited to participate in the Hot School Meals Programme from April 2024."
   "Child Benefit is a universal monthly payment made to families with children up to the age of 16 years."
   "The Dental Treatment Services Scheme (DTSS) provides dental care, free of charge, to medical card holders aged 16 and over."])

(comment
  ;; Interesting how many matches there are for this! Shows the duplication of language
  (filter #(re-find (re-pattern (str/join "|" highlights)) (second %))
          (map-indexed vector (remove nil? (:answer ds)))))


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



;; Simply splits into sentences and group sentences by 'chunk size'
(defn chunking-fn [documents chunk-size]
  (let [sentences (->> (str/split documents tokenizer/sentence-splitter-re)
                       (remove #(= (str/trim %) "")))
        chunks (->> (partition-all chunk-size sentences)
                    (mapv #(str/join #" " %)))]
    chunks))




(comment
  (chunking-fn "
The “walk in” was uttered with closed teeth, and expressed the sentiment, “Go to the Deuce!” even the gate over which he leant manifested no sympathising movement to the words; and I think that circumstance determined me to accept the invitation: I felt interested in a man who seemed more exaggeratedly reserved than myself.

When he saw my horse’s breast fairly pushing the barrier, he did put out his hand to unchain it, and then sullenly preceded me up the causeway, calling, as we entered the court,—“Joseph, take Mr. Lockwood’s horse; and bring up some wine.”

“Here we have the whole establishment of domestics, I suppose,” was the reflection suggested by this compound order. “No wonder the grass grows up between the flags, and cattle are the only hedge-cutters.”

Joseph was an elderly, nay, an old man, very old, perhaps, though hale and sinewy. “The Lord help us!” he soliloquised in an undertone of peevish displeasure, while relieving me of my horse: looking, meantime, in my face so sourly that I charitably conjectured he must have need of divine aid to digest his dinner, and his pious ejaculation had no reference to my unexpected advent. "
               3))

(def embedding-model (AllMiniLmL6V2EmbeddingModel/new))

(defn chunked-docs [docs chunking-size]
  (->> (mapv #(chunking-fn % chunking-size) docs) ;; In this case chunk the documents individually, because we know that are all separate/discrete answers
       (remove empty?)
       (reduce into)))

;; TODO: add ability to use openai embeddings

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
        (println corresponding-ans)
        (recur (inc idx) (conj res (tokenizer/calculate-retrieval-metrics
                                    (nth highlights idx)
                                    corresponding-ans
                                    :word
                                    "Question Method")))))))


(defonce met-comparisons
  (let [hls highlights-answers
        docs (-> ds
                 (tc/drop-missing :answer)
                 (tc/drop-rows #(re-find #"details supplied" (% :question)))
                 (tc/drop-rows #(re-find #"As this is a service matter" (% :answer)))
                 (tc/select-rows (range 500))
                 :answer)
        full-docs-benchmark (generate-metrics hls docs "Full Docs")]
    (loop [[x & xs] [3 5 10 15]
           result []]
      (if-not x
        (conj result full-docs-benchmark)
        (let [chunked-docs (chunked-docs docs x)]
          (recur xs
                 (conj result (generate-metrics hls chunked-docs (str x " Chunks")))))))))

(defn average [coll]
  (float
   (/ (apply + coll)
      (count coll))))



(def temp-ds
  (->
   (tc/dataset (into (reduce into met-comparisons) (generate-metrics-question-retrieval-method highlights-answers highlights-questions)))
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
