(ns notebooks.llm-api
  (:require [clj-http.client :as client]
            [jsonista.core :as json]
            [wkok.openai-clojure.api :as api]
            [clojure.edn :as edn]))

;; ### Different API calls
;;
;; #### Local
;; Local models are run on Ollama
(defn api-llm-local [model form]
  (client/post "http://localhost:11434/api/chat"
               {:form-params
                {:model model
                 :messages form
                 :stream false}
                :content-type :json}))

;; #### Openai
(defn api-llm-openai [model form]
  (api/create-chat-completion
   {:model model
    :messages form}
   {:api-key (:openai-api-key (edn/read-string (slurp "secrets.edn")))}))

;; #### Google
(defn api-llm-google [model form]
  (client/post (str "https://generativelanguage.googleapis.com/v1beta/models/"
                    model
                    ":generateContent?key="
                    (:gemini-api-key (edn/read-string (slurp "secrets.edn"))))
               {:form-params form
                :content-type :json}))

;; #### Anthropic - Claude
(defn api-llm-claude [model form]
  (client/post "https://api.anthropic.com/v1/messages"
               {:form-params
                (merge
                 {:model model
                  :max_tokens 1024}
                 form)
                :content-type :json
                :headers {:x-api-key (:anthropic-api-key (edn/read-string (slurp "secrets.edn")))
                          :anthropic-version "2023-06-01"}}))

;; Helpers to extract content from the responses.
(defn resp->body [resp] (-> resp :body (json/read-value json/keyword-keys-object-mapper)))
(defn get-content-llm-local [resp] (-> resp resp->body :message :content))
(defn get-content-llm-openai [resp] (-> resp :choices first :message :content))
(defn get-content-llm-google [resp] (-> resp resp->body :candidates first :content :parts first :text))
(defn get-content-llm-claude [resp] (-> resp resp->body  :content first :text))

;; TODO: these fns return the text content of resp. Maybe consider what other data might be relevant
(defn ask-llm-openai [{:keys [question model-ref system-prompt]}]
  (->
   (api-llm-openai model-ref
                   (if system-prompt
                     [{:role "system" :content system-prompt}
                      {:role "user" :content question}]
                     [{:role "user" :content question}]))
   get-content-llm-openai))

(defn ask-llm-google [{:keys [question model-ref system-prompt]}]
  (->
   (api-llm-google model-ref
                   (if system-prompt
                     {:system_instruction
                      {:parts [{:text system-prompt}]}
                      :contents {:parts [{:text question}]}}
                     {:contents {:parts [{:text question}]}}))
   (get-content-llm-google)))


(defn ask-llm-claude [{:keys [question model-ref system-prompt]}]
  (->
   (api-llm-claude
    model-ref
    {:system (or system-prompt "You are a responsible government official.")
     :messages [{:role "user" :content question}]})
   (get-content-llm-claude)))


(defn ask-llm-local [{:keys [question model-ref system-prompt]}]
  (->
   (api-llm-local model-ref
                  (if system-prompt
                    [{:role "system" :content system-prompt}
                     {:role "user" :content question}]
                    [{:role "user" :content question}]))
   (get-content-llm-local)))


;; ## Model References

(def llm-models
  [{:model-ref "llama3.1" :platform "Ollama" :name "Llama3.1" :parameters "8B"  :model-type "local"}
   {:model-ref "llama3.2" :platform "Ollama" :name "Llama3.2" :parameters "3B"  :model-type "local"}
   {:model-ref "mistral" :platform "Ollama" :name "Mistral" :parameters "7B"  :model-type "local"}
   {:model-ref "llava" :platform "Ollama" :name "LLaVa" :parameters "7B"  :model-type "local"}
   {:model-ref "deepseek-r1" :platform "Ollama" :name "Deepseek R1" :parameters "7B"  :model-type "local"}
   {:model-ref "gemma3:1b" :platform "Ollama" :name "Gemma 3" :parameters "1B"  :model-type "local"}
   {:model-ref "gemma3:4b" :platform "Ollama" :name "Gemma 3" :parameters "4B"  :model-type "local"}
   {:model-ref "granite3.2" :platform "Ollama" :name "Granite 3.2" :parameters "8B"  :model-type "local"}
   {:model-ref "gpt-4o-mini" :platform "OpenAI" :name "GPT-4 Mini" :parameters "? 8B"  :price-in 0.15 :price-out 0.6 :model-type "cloud"}
   {:model-ref "gpt-4o" :platform "OpenAI" :name "GPT-4o" :parameters "?"  :price-in 2.5 :price-out 10 :model-type "cloud"}
   {:model-ref "o4-mini-2025-04-16" :platform "OpenAI" :name "GPT-4o" :parameters "?"  :price-in 1.10 :price-out 4.40 :model-type "cloud"}
   {:model-ref "o3-mini" :platform "OpenAI" :name "GPT-o3 Mini" :parameters "?"  :price-in 1.10 :price-out 4.40 :model-type "cloud"}
   {:model-ref "gpt-3.5-turbo" :platform "OpenAI" :name "GPT-3.5 Turbo" :parameters "?"  :price-in 0.5 :price-out 1.5 :model-type "cloud"}
   {:model-ref "gemini-2.0-flash" :platform "Google" :name "Gemini 2.0 Flash" :parameters "?"  :model-type "cloud"}
   {:model-ref "gemini-2.0-flash-lite" :platform "Google" :name "Gemini 2.0 Flash Lite" :parameters "?"  :model-type "cloud"}
   {:model-ref "gemini-2.5-pro-exp-03-25" :platform "Google" :name "Gemini 2.5 Pro" :parameters "?"  :model-type "cloud"}
   {:model-ref "gemini-2.5-pro-preview-03-25" :platform "Google" :name "Gemini 2.5 Pro (Paid)" :parameters "?" :price-in 1.25 :price-out 10.0 :model-type "cloud"}
   {:model-ref "gemini-2.5-flash-preview-04-17" :platform "Google" :name "Gemini 2.5 Flash Preview"  :parameters "?" :price-in 0.15 :price-out 0.6 :model-type "cloud"}
   {:model-ref "claude-3-7-sonnet-20250219" :platform "Anthropic" :name "Claude 3.7 Sonnet"  :price-in 3.0 :price-out 15.0 :parameters "?" :model-type "cloud"}
   {:model-ref "claude-3-5-haiku-20241022" :platform "Anthropic" :name "Claude 3.5 Haiku"  :price-in 0.8 :price-out 4.0 :parameters "?" :model-type "cloud"}
   {:model-ref "claude-3-haiku-20240307" :platform "Anthropic" :name "Claude 3 Haiku"  :price-in 0.25 :price-out 1.25 :parameters "?" :model-type "cloud"}])

;; A wrapper function to check which api to use
;; TODO: add in system prompt as option
;;
;; usage:
;; {:model-ref "model"
;;  :question "question"
;;  :system-prompt "prompt"}
(defn ask-llm [{:keys [model-ref] :as params}]
  (let [get-models (fn [platform] (->>  llm-models
                                        (filterv #(= (:platform %) platform))
                                        (mapv :model-ref)
                                        (into #{})))]
    (condp some [model-ref]
      (get-models "Ollama")    (ask-llm-local params)
      (get-models "OpenAI")    (ask-llm-openai params)
      (get-models "Google")    (ask-llm-google params)
      (get-models "Anthropic") (ask-llm-claude params)
      nil)))
