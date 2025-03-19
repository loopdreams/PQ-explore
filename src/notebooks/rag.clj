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

(defn ask-llm [question model]
  (let [prompt (make-prompt question)]
    (-> (client/post "http://localhost:11434/api/chat"
                     {:form-params
                      {:model model
                       :messages [{:role "user" :content prompt}]
                       :stream false}
                      :content-type :json})
        :body
        (json/read-value json/keyword-keys-object-mapper))))


(defonce test-response (ask-llm "what is the government doing about about climate change?"  "llama3.1"))

(kind/md
 (:content (:message test-response)))

(defonce test-response-2 (ask-llm "What are the government's plans for legislation?"  "llama3.1"))

(kind/md
 (:content (:message test-response-2)))

(defonce test-response-3 (ask-llm "What measures are the government taking to enhance Ireland's cybersecurity?"  "llama3.1"))

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

;; ## Sentence by Sentence Similarity

(defn split-sentences [text] (remove #(= % "") (str/split text #"\. |\n")))

(split-sentences (-> test-response :message :content))

(defn response-sentence-scores [response]
  (let [sentences (split-sentences (-> response :message :content))
        data (mapv (fn [s] (let [score (similarity-test s db-store-answers)]
                             {:score score
                              :text s}))
                   sentences)]
    data))


(defn get-score-color [score]
  (condp > score
    0.5 "#F195A9"
    0.7 "#EAD196"
    0.8 "#E1EACD"
    "#BAD8B6"))


;; TODO: preserve formatting (newlines, bullets, etc)
(defn color-response-score [sentence-scores]
  (into [:p]
        (map (fn [{:keys [score text]}]
               [:span {:style (str "background-color: " (get-score-color score))} text])
             sentence-scores)))

(kind/hiccup
 (color-response-score (response-sentence-scores test-response)))

(kind/hiccup
 (color-response-score (response-sentence-scores test-response-2)))

(kind/hiccup
 (color-response-score (response-sentence-scores test-response-3)))

;; ## Comparing a question without context


(defn ask-llm-no-context [question model]
  (-> (client/post "http://localhost:11434/api/chat"
                   {:form-params
                    {:model model
                     :messages [{:role "user" :content question}]
                     :stream false}
                    :content-type :json})
      :body
      (json/read-value json/keyword-keys-object-mapper)))

(defonce test-response-4 (ask-llm-no-context "what is the Irish government doing about about climate change?"  "llama3.1"))

(similarity-test (-> test-response-4 :message :content) db-store-answers)

(kind/hiccup
 (color-response-score (response-sentence-scores test-response-4)))

(defonce test-response-5 (ask-llm-no-context "What measures are the government taking to enhance Ireland's cybersecurity?"  "llama3.1"))


(similarity-test (-> test-response-5 :message :content) db-store-answers)

(kind/hiccup
 (color-response-score (response-sentence-scores test-response-5)))

;; Seems like the llm is already quite informed about Ireland's policies!
;;
;; ## Comparing Responses from Weaker Models
;;
;; Let's try some responses from a 1B parameter model (gemma 3)


(defonce test-response-6 (ask-llm "what is the government doing about about climate change?" "gemma3:1b"))
(defonce test-response-7 (ask-llm-no-context "what is the Irish government doing about about climate change?" "gemma3:1b"))

(kind/md
 (-> test-response-6 :message :content))

(kind/md
 (-> test-response-7 :message :content))

(similarity-test (-> test-response-6 :message :content) db-store-answers)
(similarity-test (-> test-response-7 :message :content) db-store-answers)

(kind/hiccup
 (color-response-score (response-sentence-scores test-response-6)))

(kind/hiccup
 (color-response-score (response-sentence-scores test-response-7)))
