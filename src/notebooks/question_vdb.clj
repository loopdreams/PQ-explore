;; # Questions Vector Database
(ns notebooks.question-vdb
  (:require [tablecloth.api :as tc]
            [notebooks.preparation :refer [ds]]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.tableplot.v1.plotly :as plotly]
            [clojure.string :as str])
  (:import (dev.langchain4j.data.embedding Embedding)
           (dev.langchain4j.data.segment TextSegment)
           (dev.langchain4j.model.embedding EmbeddingModel)
           (dev.langchain4j.store.embedding EmbeddingMatch)
           (dev.langchain4j.store.embedding.inmemory InMemoryEmbeddingStore)
           (dev.langchain4j.model.embedding.onnx.allminilml6v2 AllMiniLmL6V2EmbeddingModel)))

;; ## Introduction [Intro text]


;; ## Building a Vector Database from the Questions

;; We will first use langchain4j to transform question text into vector embeddings. The model
;; we will use is provided by langchain4j.

(def model (AllMiniLmL6V2EmbeddingModel/new))

(def questions-list (:question ds))

;; Let's look at a single embedding for reference.

(->> (TextSegment/from (first questions-list))
     (. model embed)
     (.content))

;; As you can see, it is a very long vector of floating points representing the text.
;;
;; In order to store the embeddings in a database, we will use an in-memory database provided
;; by langchain4j. There are also options for using a more robust solution like postgres, but
;; for testing/exploration purposes, an in-memory database will do.

(def db-store (InMemoryEmbeddingStore/new))

;; A short function for adding a question to the store.

(defn add-question-to-store! [question store]
  (let [segment (TextSegment/from question)
        embedding (->> segment (. model embed) (.content))]
    (. store add embedding segment)))

;; Finally, adding all the questions to the store. The mapping, which has side effects, is wrapped in a 'count' function
;; just so we can see at the end how many questions have been added.

(count (map #(add-question-to-store! % db-store) questions-list))

;; ## Testing Lookup

;; Next, let's try to use the database to return similar questions from the database.

(def test-question (first questions-list))

test-question

;; It's a very generic question...

(defn query-db-store [text n]
  (let [query (.content (. model embed text))
        result (. db-store findRelevant query n)]
    (map (fn [entry]
           {:text (.text (.embedded entry))
            :score (.score entry)})
         result)))

(kind/table
 (query-db-store test-question 5))

;; Let's try with a made-up question

(kind/table
 (query-db-store "The Deputy would like to ask the Minister about schemes relating to climate change." 5))

;; Finally, let's also try with a more focused question

(-> ds
    (tc/select-rows #(= (% :topic) "Dublin Bus"))
    :question
    first
    kind/md)

(def test-question-2 "The Deupty asked the Minister about Dublin Bus Route 74 and what their plans for this route are.")

(kind/table
 (query-db-store test-question-2 5))

;; Two of the five questions relate to the national bus service (Bus Eireann) as opposed to Dublin Bus, but overall
;; it is quite close.

;; ## Answering a Question
;;
;; Let's finally use this method to return the best answers based on the question given. This
;; approach could be later used to provide context for a RAG model that would generate answers.

(defn get-answers-for-question [question]
  (let [matching-questions (map :text (query-db-store question 5))]
    (-> ds
        (tc/select-rows #(some #{(% :question)} matching-questions))
        (tc/select-columns [:answer :date]))))

(kind/table
 (get-answers-for-question test-question-2))


;; Using just chatgpt, here is how a sample answer might possibly be generated:
;;
;; > The Minister stated that while he is responsible for public transport policy and funding, the day-to-day operations, including Dublin Bus Route 74, fall under the National Transport Authority (NTA). He has therefore referred the Deputy’s question to the NTA for a direct reply.

;; ## Question Similarity Across the Database
;;
;; For this exploration, we will take the first 100 questions, and count how
;; many questions are in the database based on similarity

(def test-questions (take 100 questions-list))


(defn count-similar-questions-above-threshold [question threshold]
  (let [candidates (query-db-store question 10000)
        relevant (filter (fn [{:keys [score]}] (> score threshold)) candidates)]
    (- (count relevant) 1)))

(-> {:n-similar (map #(count-similar-questions-above-threshold % 0.9) test-questions)}
    (tc/dataset)
    (plotly/layer-histogram {:=x :n-similar}))


(-> {:n-similar (map #(count-similar-questions-above-threshold % 0.85) test-questions)}
    (tc/dataset)
    (plotly/layer-histogram {:=x :n-similar}))

(-> {:n-similar (map #(count-similar-questions-above-threshold % 0.75) test-questions)}
    (tc/dataset)
    (plotly/layer-histogram {:=x :n-similar}))

;; At a threshold of 0.75, 6 of the sample questions have more than 600 other similar questions
;;
;; Let's look at these questions

(filter (fn [question] (> (count-similar-questions-above-threshold question 0.75) 600))
        test-questions)

;; They all relate to Education/schools

;; Let's look at the two questions that have greater than 14 similar questions with a 90% threshold

(filter (fn [question] (> (count-similar-questions-above-threshold question 0.9) 14))
        test-questions)

;; ### Similarity by Topic

;; We'll look at the top 7 main topics

(def top-7-topics
  (-> (tc/group-by ds [:topic])
      (tc/aggregate {:count tc/row-count})
      (tc/order-by :count [:desc])
      (tc/select-rows (range 7))))

top-7-topics

;; Next, let's try to write a function to count the level of similarity among a topic group.
;;
;; The idea here is to:
;;
;; 1. Create a mini-store of all the questions related to a topic
;;
;; 2. Check each question against this store, and count it if it has at least 1 (excluding itself) question that is similar to it, based on a threshold
;;
;; 3. Calculate the percentage of questions that have at least one other similar question

(defn percentage-similar-questions [store questions threshold]
  (let [qs-above-threshold
        (reduce (fn [res question]
                  (let [query (.content (. model embed question))
                        matches (. store findRelevant query 100)]
                    (conj res
                          (-> (filter (fn [match] (> (.score match) threshold)) matches)
                              count
                              (- 1)))))
                [] questions)
        count-qs-above-threshold (count (remove zero? qs-above-threshold))]
    (float (/ count-qs-above-threshold (count questions)))))


(defn category-similarity [name type threshold]
  (let [target-questions (-> ds (tc/select-rows #(= (type %) name)) :question)
        topic-store (InMemoryEmbeddingStore/new)
        add-question (fn [q]
                       (let [seg (TextSegment/from q)
                             emb (->> seg (. model embed) (.content))]
                         (. topic-store add emb seg)))]
    (do
     (mapv add-question target-questions)
     (percentage-similar-questions topic-store target-questions threshold))))


(let [topics (:topic top-7-topics)
      data (mapv (fn [topic] (let [score (category-similarity topic :topic 0.9)]
                               {:topic topic
                                :score score}))
                 topics)]
  (-> (tc/dataset data)
      (plotly/layer-bar
       {:=x :topic
        :=y :score})))

;; At a lower threshold the topics are highly similar:

(let [topics (:topic top-7-topics)
      data (mapv (fn [topic] (let [score (category-similarity topic :topic 0.75)]
                               {:topic topic
                                :score score}))
                 topics)]
  (-> (tc/dataset data)
      (plotly/layer-bar
       {:=x :topic
        :=y :score})))


;; Based on the above, of the top 7 topics, the 'Health Services' questions seem the most diverse, which makes sense.

;; We can do a similar kind of comparison with the top seven departments

(def top-7-departments
  (-> ds
      (tc/group-by [:department])
      (tc/aggregate {:count tc/row-count})
      (tc/order-by :count :desc)
      (tc/select-rows (range 7))))

top-7-departments

;; (let [departments (:department top-7-departments)
;;       data (mapv (fn [department] (let [score (category-similarity department :department 0.9)]
;;                                     {:department department
;;                                      :score score}))
;;                  departments)]
;;   (-> (tc/dataset data)
;;       (plotly/layer-bar
;;        {:=x :department
;;         :=y :score})))

;; How about some example questions that aren't similar to other questions?

(defn outlier-questions
  "Looks for the most similar questions (excluding self) and checks if score is below threshold."
  [topic threshold]
  (let [target-questions (-> ds (tc/select-rows #(= (:topic %) topic)) :question)]
    (filter (fn [question]
              (let [query (.content (. model embed question))
                    match (. db-store findRelevant query 2)]
                (< (.score (second match)) threshold)))
            target-questions)))


(->> 0.80
     (outlier-questions "Disability Services")
     (str/join "\n\n")
     kind/md)

(->> 0.77
     (outlier-questions "Health Services")
     (str/join "\n\n")
     kind/md)

(->> 0.82
     (outlier-questions "Schools Building Projects")
     (str/join "\n\n")
     kind/md)


;; Question Similarity Curves


;; (defn category-similarity [name type threshold]
;;   (let [target-questions (-> ds (tc/select-rows #(= (type %) name)) :question)
;;         topic-store (InMemoryEmbeddingStore/new)
;;         add-question (fn [q]
;;                        (let [seg (TextSegment/from q)
;;                              emb (->> seg (. model embed) (.content))]
;;                          (. topic-store add emb seg)))]
;;     (do
;;      (mapv add-question target-questions)
;;      (percentage-similar-questions topic-store target-questions threshold))))

(def db-store-ip (InMemoryEmbeddingStore/new))

(let [ip-questions (:question (tc/select-rows ds #(= (:topic %) "International Protection")))]
  (count
   (mapv (fn [q]
           (let [seg (TextSegment/from q)
                 emb (->> seg (. model embed) (.content))]
             (. db-store-ip add emb seg)))
         ip-questions)))

(def test-question-3 (first (:question (tc/select-rows ds #(= (:topic %) "International Protection")))))

(defn all-similarity-scores [question store & ignore-self?]
  (let [emb (.content (. model embed question))
        result (. store findRelevant emb 200)]
    (map-indexed (fn [idx e] {:idx idx :score (.score e)})
                 (if ignore-self? (rest result) result))))

(-> (all-similarity-scores test-question-3 db-store-ip true)
    (tc/dataset)
    (plotly/layer-line
     {:=x :idx
      :=y :score}))

(defn plot-n-random-scores [topic store n]
  (let [candidates (:question (tc/select-rows ds #(= (:topic %) topic)))
        selected (take n (repeatedly #(rand-nth candidates)))
        data (loop [n 0 result []]
               (if (= n (count selected)) result
                   (let [q (nth selected n)
                         scores (all-similarity-scores q store true)
                         scores (map #(assoc % :question n) scores)]
                     (recur (inc n) (into result scores)))))]
    (tc/dataset data)))

(-> (plot-n-random-scores "International Protection" db-store-ip 1)
    (plotly/base
     {:=x :idx
      :=y :score})
    plotly/layer-point
    plotly/layer-smooth)

(-> (plot-n-random-scores "International Protection" db-store-ip 20)
    (plotly/base
     {:=x :idx
      :=y :score})
    plotly/layer-point
    plotly/layer-smooth)

(-> (plot-n-random-scores "International Protection" db-store-ip 50)
    (plotly/base
     {:=x :idx
      :=y :score})
    plotly/layer-point
    plotly/layer-smooth)

(-> (plot-n-random-scores "International Protection" db-store-ip 50)
    (plotly/layer-histogram
     {:=x :score}))

(-> (plot-n-random-scores "International Protection" db-store-ip 1)
    (plotly/layer-histogram
     {:=x :score}))

;; For comparison, let's try an "International Protection" question
;; within a different topic area


(def db-store-schools (InMemoryEmbeddingStore/new))

(let [schools-questions (:question (tc/select-rows ds #(= (:topic %) "Schools Building Projects")))]
  (count
   (mapv (fn [q]
           (let [seg (TextSegment/from q)
                 emb (->> seg (. model embed) (.content))]
             (. db-store-schools add emb seg)))
         schools-questions)))

(def random-ip-question (rand-nth (:question (tc/select-rows ds #(= (:topic %) "International Protection")))))

(def scores-self-group-other-group
  (let [scores-ip (all-similarity-scores random-ip-question db-store-ip true)
        scores-ip (map (fn [s] (assoc s :batch "same-group")) scores-ip)
        scores-sch (all-similarity-scores random-ip-question db-store-schools)
        scores-sch (map (fn [s] (assoc s :batch "other-group")) scores-sch)]
    (-> (concat scores-ip scores-sch)
        (tc/dataset))))

(-> scores-self-group-other-group
    (plotly/base
     {:=x :idx
      :=y :score
      :=color :batch})
    (plotly/layer-point)
    (plotly/layer-smooth))

(-> scores-self-group-other-group
    (plotly/layer-histogram
     {:=x :score
      :=color :batch
      :=mark-opacity 0.7}))
