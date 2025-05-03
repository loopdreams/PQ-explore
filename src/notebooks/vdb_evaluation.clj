;; # Retrieval Evaluation
(ns notebooks.vdb-evaluation
  (:require [clojure.edn :as edn]
            [notebooks.preparation :refer [ds]]
            [notebooks.vector-database :refer [add-doc-to-store db-store-questions query-db-store]]
            [scicloj.kindly.v4.kind :as kind]
            [notebooks.tokenizer :as tokenizer]
            [scicloj.tableplot.v1.plotly :as plotly]
            [clojure.string :as str]
            [tablecloth.api :as tc]
            [notebooks.vdb-evaluation :as vdb]
            [jsonista.core :as json])
  (:import
   (dev.langchain4j.data.segment TextSegment)
   (dev.langchain4j.store.embedding.inmemory InMemoryEmbeddingStore)
   (dev.langchain4j.model.embedding.onnx.allminilml6v2 AllMiniLmL6V2EmbeddingModel)))

;; In this section we'll use some standard approaches to try to evaluate the
;; performance of the vector database.
;;
;; The material fetched from the database makes up the context that is provided
;; in a RAG application, so it is important to try get information that is both
;; **relevant to the question**, but also **doesn't contain too much superfluous
;; information** that might confuse the LLM or make the context information
;; ambiguous.
;;
;;
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

(def highlights-questions
  ["What is the government doing to help improve GP services?"
   "Will the government put in place Level 6 (QQI) courses for healthcare assistants?"
   "What is the government doing with regard to the National Drugs Strategy?"
   "How is the government encouraging local authorities to apply for the town and village renewal scheme?"
   "What is the salary scale for an archaeologist in the local government sector?"])

;;
;; ## Evaluating Retrieval
;; At the moment, our approach is based on *searching for similar questions*,
;; and then returning their answers.  However, this is a very naive
;; approach. It is based on the concrete steps that are usually taken when trying to answer a new question:
;;
;; 1. Search for previous similar questions
;;
;; 2. Scan these answers for relevant info
;;
;; Why not just skip this step of searching through questions all together? The
;; approach chosen here might depend on the actual application of a system like
;; this. For example, would the intended use be to help administers prepare
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
;; too much information, and could potentially confuse or mislead the LLM
;; later on.
;;
;; To help improve this, let's first take a look at some common, simple metrics
;; that are used to measure retrieval
;; ### Retrieval Metrics Overview

;; #### Recall
;;
;; Recall measures how much of the ground truth statement is captured through
;; the retrieval. It does so by counting how many tokens in the retrieved context
;; are also present in the ground truth statement.
;;
;; Higher recall means that more of the ground truth statement is captured in the context.
;;
;; #### Precision
;; Precision is like the inverse of recall, it counts how many tokens in the
;; ground-truth statement are present in the retrieved context.
;;
;; Higher precision means that there is less 'additional' information captured in the context.
;;
;; #### Intersection over Union (IoU)
;; This is a metric that captures both recall and precision together. It is
;; calculated by dividing the length of the token overlap by the length of token
;; union of the retrieved context and the ground truth.
;;
;; For example, if we retrieve 200 tokens, the ground truth is 100 tokens, and
;; the overlap between both is 70 tokens, then the IoU would be 70/(200+100-70)
;; = 0.304

;; ### Splitting the Documents

;; When thinking about recall/precision, a natural question that we might ask
;; relates to the size of the documents that are stored in the database. For
;; example, if we are always retrieving the 5 most similar docs, but each of
;; those are 1000 words long, then we are always passing the model 5000 words.
;; This will probably increase the chance of the target information being
;; present, but it will also increase the amount of superfluous information
;; passed to the LLM (i.e., decrease precision)
;;
;; Therefore, it might be useful to try splitting the documents into chunks of
;; various sizes and see which size works best.
;;
;; Below is a simple function that does this for us.
(defn split-document [documents chunk-size]
  (let [sentences (->> (str/split documents tokenizer/sentence-splitter-re)
                       (remove #(= (str/trim %) "")))
        chunks    (->> (partition-all chunk-size sentences)
                       (mapv #(str/join #" " %)))]
    chunks))

;; ### Calculating Metrics for Retrieval Strategies
;;
;;
;; To test our retrieval strategies, we will split up the information contained in the
;; dataset 'answers' in the following ways:
;;
;; - Chunks of 3 sentences (Chunks 3)
;;
;; - Chunks of 5 sentences (Chunks 5)
;;
;; - Chunks of 10 sentences (Chunks 10)
;;
;; - Chunks of 15 sentences (Chunks 15)
;;
;; - Original answers, i.e., no splitting (Full Docs)
;;
;; We will also test our original 'Question Retrieval' method that we created in
;; the last section (Question Method).

(def embedding-model (AllMiniLmL6V2EmbeddingModel/new))

(defn chunked-docs [docs chunking-size]
  (->> (mapv #(split-document % chunking-size) docs) ;; In this case chunk the documents individually, because we know that are all separate/discrete answers
       (remove empty?)
       (reduce into)))

;; To generate metrics, we first retrieve similar documents based on a given
;; question, then we check those documents against the highlighted (answer)
;; text.
;;
;; Both functions below are mostly identical, except the 'question method'
;; function adds a step from looking up an answer based on a retrieved similar
;; question, it also re-used the previously created db-store-questions, so it's
;; much quicker to run.

(defn calculate-metrics [questions answers chunked-docs & label]
  (let [db-store (InMemoryEmbeddingStore/new)
        num      (count (mapv #(add-doc-to-store % db-store) chunked-docs))
        _        (println num)]
    (loop [idx     (dec (count questions))
           results []]
      (if (< idx 0)
        results
        (let [q-embedding (->> (TextSegment/from (nth questions idx))
                               (. embedding-model embed)
                               (.content))
              matches     (->> (. db-store findRelevant q-embedding 5)
                               (mapv #(.text (.embedded %)))
                               (str/join " "))]
          (recur (dec idx)
                 (conj results (tokenizer/calculate-retrieval-metrics
                                (nth answers idx)
                                matches
                                :word
                                (first label)))))))))

(defn calculate-metrics-question-retrieval-method [hl-answers hl-questions]
  (loop [idx 0
         res []]
    (if (= idx (count hl-answers))
      res
      (let [q-embedding       (->> (TextSegment/from (nth hl-questions idx))
                                   (. embedding-model embed)
                                   (.content))
            q-matches         (->> (. db-store-questions findRelevant q-embedding 5)
                                   (mapv #(.text (.embedded %))))
            corresponding-ans (->> (tc/select-rows ds #(some #{(:question %)} q-matches))
                                   :answer
                                   (str/join " "))]
        (recur (inc idx)
               (conj res (tokenizer/calculate-retrieval-metrics
                          (nth hl-answers idx)
                          corresponding-ans
                          :word
                          "Question Method")))))))


;; Running these functions over several types of chunking strategies takes a
;; while, so we are going to run them and save the results to a file.
(comment
  (defonce metric-comparisons
    (let [questions           highlights-questions
          answers             highlights-answers
          docs                (-> ds
                                  (tc/drop-missing :answer)
                                  (tc/drop-rows #(re-find #"details supplied" (% :question))) ;; getting rid of a few extra unhelpful questions/answers
                                  (tc/drop-rows #(re-find #"As this is a service matter" (% :answer)))
                                  :answer)
          full-docs-benchmark (calculate-metrics questions answers docs "Full Docs")]
      (loop [[x & xs] [3 5 10 15]
             result   []]
        (if-not x
          (conj result full-docs-benchmark)
          (let [chunked-docs (chunked-docs docs x)]
            (recur xs
                   (conj result (calculate-metrics questions answers chunked-docs (str "Chunks-" x)))))))))


  (spit "data/retrieval_metrics/results.edn"
        (with-out-str (clojure.pprint/pprint
                       (into (reduce into metric-comparisons)
                             (calculate-metrics-question-retrieval-method highlights-answers highlights-questions))))))

(def comparison-data (edn/read-string (slurp "data/retrieval_metrics/results.edn")))

(kind/table comparison-data)

(defn average [coll]
  (float
   (/ (apply + coll)
      (count coll))))

(def ds-metrics-avg
  (->
   (tc/dataset comparison-data)
   (tc/group-by [:label])
   (tc/aggregate {:avg-recall #(average (% :recall))
                  :avg-precision #(average (% :precision))
                  :avg-IoU #(average (% :IoU))})))

(-> ds-metrics-avg
    (plotly/layer-line
     {:=x :label
      :=y :avg-recall}))


(-> ds-metrics-avg
    (plotly/layer-line
     {:=x :label
      :=y :avg-precision}))


(-> ds-metrics-avg
    (plotly/layer-line
     {:=x :label
      :=y :avg-IoU}))


;; The results are mostly as expected - with larger chunk sizes recall goes up,
;; although it notably dips with chunk sizes of 10. At the same time, precision
;; goes down - there is more excess information that is perhaps not needed.
;;
;; Interestingly, the 'question retrieval' method has slightly higher precision
;; than the method of looking up the full docs, even though in both cases what
;; is returned are full answers. This suggests that perhaps the 'naive' approach
;; of searching for similar questions and returning the corresponding answers is
;; perhaps slightly more precise that searching through the full answers.
;;
;; Still, it looks like using either chunks of 3 or 5 sentences might be the best overall.
;; For the RAG application, we'll try using chunks of 3 sentences.
;;
;; Let's build that database and save it to a file.

(comment
  (let [answers (-> ds
                    (tc/drop-missing :answer)
                    (tc/drop-rows #(re-find #"details supplied" (% :question)))
                    (tc/drop-rows #(re-find #"As this is a service matter" (% :answer)))
                    :answer)

        docs (-> (chunked-docs answers 3)
                 distinct) ;; before filtering for duplicates there were around 24K chunks, after filtering around 18K
        db-store (InMemoryEmbeddingStore/new)
        _c (count (mapv #(add-doc-to-store % db-store) docs))]
    (println _c)
    (spit "data/retrieval_store/db-store-docs.json" (.serializeToJson db-store))))


(def db-store-chunked-answers (InMemoryEmbeddingStore/fromFile "data/retrieval_store/db-store-docs.json"))


;; ### Final Checks
;;
;; Let's have a look at the context that is actually generated by each of the
;; approaches to see the difference that the alternate retrieval strategy can
;; make.

(defn gererate-context-question-retrieval [question]
  (let [related-questions (->> (query-db-store question 5)
                               (mapv :text))
        past-answers (-> ds
                         (tc/select-rows #(some #{(:question %)} related-questions))
                         :answer)]
    (mapv (fn [a] {:text a}) past-answers)))

(defn generate-context [question db-store-name]
  (if (= db-store-name :question-retrieval)
    (gererate-context-question-retrieval question)
    (let [emb-question (.content (. embedding-model embed question))
          related-docs (. db-store-chunked-answers findRelevant emb-question 5)]
      (map (fn [doc]
             {:text (.text (.embedded doc))
              :score (.score doc)})
           related-docs))))



;; First we'll look at a very general question about GP services:

(kind/table
 (generate-context "What is the government doing to help improve GP services?" :db-docs))

;; These answers are not a bad starting point for answering this kind of broad
;; question.
;;
;; Looking at the first answer in the table above, the figure of '211m EUR' is
;; referenced in relation to the 2019 GP Agreement. Let's see if the database
;; can match this exact figure:

(kind/table
 (generate-context "How much annual investment was provided under the 2019 GP agreement?" :db-docs))

;; Every document retrieved seems to contain the relevant figure :)
;;
;; For completness, let's try this same, more specific, question with the
;; previous approach. As you can see below It's much less focused!


(kind/table
 (generate-context "How much annual investment was provided under the 2019 GP agreement" :question-retrieval))
