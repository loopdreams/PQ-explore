;; # Questions Vector Database
(ns notebooks.question-vdb
  (:require [tablecloth.api :as tc]
            [notebooks.preparation :refer [ds]]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.tableplot.v1.plotly :as plotly])
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

(defn add-question-to-store! [question]
  (let [segment (TextSegment/from question)
        embedding (->> segment (. model embed) (.content))]
    (. db-store add embedding segment)))

;; Finally, adding all the questions to the store. The mapping, which has side effects, is wrapped in a 'count' function
;; just so we can see at the end how many questions have been added.

(count (map add-question-to-store! questions-list))

;; ## Testing Lookup

;; Next, let's try to use the database to return a similar question.

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

(:topic ds)

(-> ds
    (tc/select-rows #(= (% :topic) "Dublin Bus"))
    :question
    first)

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

;; Let's look at the two questions that have greater than 15 similar questions with a 90% threshold

(filter (fn [question] (> (count-similar-questions-above-threshold question 0.9) 15))
        test-questions)

;; Let's narrow the focus to a single area (Education)

(def sports-funding-questions
  (-> ds
      (tc/select-rows #(= (% :topic) "Sports Funding"))
      :question))


(-> {:n-similar (map #(count-similar-questions-above-threshold % 0.89) sports-funding-questions)}
    (tc/dataset)
    (plotly/layer-histogram {:=x :n-similar}))
