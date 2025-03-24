;; # RAG Test
(ns notebooks.rag
  (:require [notebooks.question-vdb :as q]
            [notebooks.preparation :refer [ds]]
            [clj-http.client :as client]
            [scicloj.kindly.v4.kind :as kind]
            [tablecloth.api :as tc]
            [scicloj.tableplot.v1.plotly :as plotly]
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

(defn make-context-prompt [question]
  (let [ctx (build-context question)]
    (str "You are a responsible government official. Provide an informative, short answer to the user's question, using the supplied context only."
         " Context: " ctx)))

(defn ask-llm-local [question model]
  (let [prompt (make-context-prompt question)]
    (-> (client/post "http://localhost:11434/api/chat"
                     {:form-params
                      {:model model
                       :messages [{:role "system" :content prompt}
                                  {:role "user" :content question}]
                       :stream false}
                      :content-type :json})
        :body
        (json/read-value json/keyword-keys-object-mapper))))

(def local-llm-models
  [{:name "Llama3.1" :parameters "8B" :model-ref "llama3.1"}
   {:name "Llama3.1" :parameters "3B" :model-ref "llama3.2"}
   {:name "Mistral" :parameters "7B" :model-ref "mistral"}
   {:name "LLaVa" :parameters "7B" :model-ref "llava"}
   {:name "Deepseek R1" :parameters "7B" :model-ref "deepseek-r1"}
   {:name "Gemma 3" :parameters "1B" :model-ref "gemma3:1b"}
   {:name "Gemma 3" :parameters "4B" :model-ref "gemma3"}
   {:name "Granite 3.2" :parameters "8B" :model-ref "granite3.2"}])

(kind/table (sort-by :name local-llm-models))

(defn unix-timestamp []
  (str (quot (System/currentTimeMillis) 1000)))

(defn run-models-and-write-answers-to-disk! [models question]
  (spit (str "data/" (unix-timestamp) "-model_responses.edn")
        (reduce (fn [res {:keys [model-ref] :as model}]
                  (let [answer (-> (ask-llm-local question model-ref)
                                   :message
                                   :content)]
                    (conj res
                          (assoc model :response answer))))
                []
                models)))

(comment
  (time
   (run-models-and-write-answers-to-disk!
    (take 2 local-llm-models)
    "Deputy Emer Currie asked the Minister for Transport his plans to expand EV charging points at the State's airports to facilitate more EV drivers and an increase in EV car rental; and if he will make a statement on the matter.")))


(defonce model-responses (clojure.edn/read-string (slurp "data/1742847936-model_responses.edn")))

(def db-store-answers (InMemoryEmbeddingStore/new))

(def answers-list
  (-> ds
      (tc/drop-missing :answer)
      :answer))

(count (map #(q/add-question-to-store! (str %) db-store-answers) answers-list))


(defn add-similarity-scores [model-data store]
  (let [query  (.content (. q/model embed (:response model-data)))
        result (. store findRelevant query 10)
        scores (mapv #(.score %) result)
        max    (first scores)
        min    (last scores)
        avg    (float (/ (apply + scores) (count scores)))]
    (-> model-data
        (assoc :max max)
        (assoc :min min)
        (assoc :avg avg))))

(def model-scores
  (mapv #(add-similarity-scores % db-store-answers) model-responses))

(kind/table
 (-> (tc/dataset model-scores)
     (tc/select-columns [:model-ref :parameters :response])))

(defn plot-model-scores [model-data]
  (-> (tc/dataset model-data)
      (plotly/base
       {:=x :model-ref})
      (plotly/layer-bar
       {:=y :min})
      (plotly/layer-bar
       {:=y :avg})
      (plotly/layer-bar
       {:=y :max})))

(plot-model-scores model-scores)








(defonce test-response (ask-llm-local "what is the government doing about about climate change?"  "llama3.1"))

(kind/md
 (:content (:message test-response)))

(defonce test-response-2 (ask-llm-local "What are the government's plans for legislation?"  "llama3.1"))

(kind/md
 (:content (:message test-response-2)))

(defonce test-response-3 (ask-llm-local "What measures are the government taking to enhance Ireland's cybersecurity?"  "llama3.1"))

(kind/md
 (:content (:message test-response-3)))


;; ## Quick test of answers


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
    0.5 "#F195A9" ;; red
    0.7 "#EAD196" ;; yellow
    0.8 "#E1EACD" ;; light green
    "#BAD8B6"))     ;; dark green



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


(defonce test-response-6 (ask-llm-local "what is the government doing about about climate change?" "gemma3:1b"))
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

;; Tasks:
;; TODO: Test Various Models
;; TODO: Ask another model to evaluate answer
;; TODO: Set context properly
