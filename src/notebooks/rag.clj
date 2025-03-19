;; # RAG Test
(ns notebooks.rag
  (:require [notebooks.question-vdb :as q]
            [notebooks.preparation :refer [ds]]
            [clj-http.client :as client]
            [scicloj.kindly.v4.kind :as kind]
            [tablecloth.api :as tc]
            [clojure.string :as str]
            [jsonista.core :as json])
  (:import (dev.langchain4j.data.embedding Embedding)
           (dev.langchain4j.data.segment TextSegment)
           (dev.langchain4j.model.embedding EmbeddingModel)
           (dev.langchain4j.store.embedding EmbeddingMatch)
           (dev.langchain4j.store.embedding.inmemory InMemoryEmbeddingStore)
           (dev.langchain4j.model.embedding.onnx.allminilml6v2 AllMiniLmL6V2EmbeddingModel)))

;; A short test for demonstrating how to provide a
;; LLM with context from existing similar questions.
;;
;; For this test, I was running an Ollama instance of llama3.1 (8B parameters) locally.

(defn build-context [question]
  (let [similar-questions (map :text (q/query-db-store question 5))
        previous-answers (-> ds
                             (tc/select-rows #(some #{(% :question)} similar-questions))
                             :answer)]
    (str/join "\n" previous-answers)))

(defn make-prompt [question]
  (let [ctx (build-context question)]
    (str "You are a responsible government official. Provide an informative, short answer to the following question, using the supplied context only."
         " Question: " question
         " Context: " ctx)))

(defn ask-llm [question]
  (let [prompt (make-prompt question)]
    (-> (client/post "http://localhost:11434/api/chat"
                     {:form-params
                      {:model "llama3.1"
                       :messages [{:role "user" :content prompt}]
                       :stream false}
                      :content-type :json})
        :body
        (json/read-value json/keyword-keys-object-mapper))))


(defonce test-response (ask-llm "what is the government doing about about climate change?"))

(kind/md
 (:content (:message test-response)))

(defonce test-response-2 (ask-llm "What are the government's plans for legislation?"))

(kind/md
 (:content (:message test-response-2)))

(defonce test-response-3 (ask-llm "What measures are the government taking to enhance Ireland's cybersecurity?"))

(kind/md
 (:content (:message test-response-3)))

;; ## Quick test of answers

(def db-store-answers (InMemoryEmbeddingStore/new))

(def answers-list
  (-> ds
      (tc/drop-missing :answer)
      :answer))

(count (map #(q/add-question-to-store! (str %) db-store-answers) answers-list))


;; A rough metric for similarity:
;;
;; - Get 10 most similar results
;;
;; - Average their scores

(defn similarity-test [text store]
  (let [query (.content (. q/model embed text))
        result (. store findRelevant query 10)
        scores (mapv (fn [r] (.score r)) result)]
    (float (/ (reduce + scores) (count scores)))))

(similarity-test "what is the government doing about about climate change?" q/db-store)

(similarity-test (:content (:message test-response)) db-store-answers)

(similarity-test "What are the government's plans for legislation?" q/db-store)

(similarity-test (:content (:message test-response-2)) db-store-answers)

(similarity-test "What measures are the government taking to enhance Ireland's cybersecurity?" q/db-store)

(similarity-test (:content (:message test-response-3)) db-store-answers)
