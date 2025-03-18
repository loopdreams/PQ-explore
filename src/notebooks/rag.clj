(ns notebooks.rag
  (:require [notebooks.question-vdb :refer [query-db-store]]
            [notebooks.preparation :refer [ds]]
            [clj-http.client :as client]
            [scicloj.kindly.v4.kind :as kind]
            [tablecloth.api :as tc]
            [clojure.string :as str]
            [jsonista.core :as json]))


(defn build-context [question]
  (let [similar-questions (map :text (query-db-store question 5))
        previous-answers (-> ds
                             (tc/select-rows #(some #{(% :question)} similar-questions))
                             :answer)]
    (str/join "\n" previous-answers)))

(defn make-prompt [question]
  (let [ctx (build-context question)]
    (str "You are a responsible government official. Provide an informative, short answer to the following question, using the supplied context only. "
         "Question: " question
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
