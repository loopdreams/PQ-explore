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
  [{:platform "Ollama" :name "Llama3.1" :parameters "8B" :model-ref "llama3.1" :model-type "local"}
   {:platform "Ollama" :name "Llama3.2" :parameters "3B" :model-ref "llama3.2" :model-type "local"}
   {:platform "Ollama" :name "Mistral" :parameters "7B" :model-ref "mistral" :model-type "local"}
   {:platform "Ollama" :name "LLaVa" :parameters "7B" :model-ref "llava" :model-type "local"}
   {:platform "Ollama" :name "Deepseek R1" :parameters "7B" :model-ref "deepseek-r1" :model-type "local"}
   {:platform "Ollama" :name "Gemma 3" :parameters "1B" :model-ref "gemma3:1b" :model-type "local"}
   {:platform "Ollama" :name "Gemma 3" :parameters "4B" :model-ref "gemma3" :model-type "local"}
   {:platform "Ollama" :name "Granite 3.2" :parameters "8B" :model-ref "granite3.2" :model-type "local"}
   {:platform "OpenAI" :name "GPT-4 Mini" :parameters "? 8B" :model-ref "gpt-4o-mini" :price-in 0.15 :price-out 0.6 :model-type "cloud"}
   {:platform "OpenAI" :name "GPT-4o" :parameters "?" :model-ref "gpt-4o" :price-in 2.5 :price-out 10 :model-type "cloud"}
   {:platform "OpenAI" :name "GPT-o3 Mini" :parameters "?" :model-ref "gpt-o3-mini" :price-in 1.10 :price-out 4.40 :model-type "cloud"}
   {:platform "OpenAI" :name "GPT-3.5 Turbo" :parameters "?" :model-ref "gpt-3.5-turbo" :price-in 0.5 :price-out 1.5 :model-type "cloud"}
   {:platform "Google" :name "Gemini 2.0 Flash" :parameters "?" :model-ref "gemini-2.0-flash" :model-type "cloud"}
   {:platform "Google" :name "Gemini 2.0 Flash Lite" :parameters "?" :model-ref "gemini-2.0-flash-lite" :model-type "cloud"}
   {:platform "Google" :name "Gemini 2.5 Pro" :parameters "?" :model-ref "gemini-2.5-pro-exp-03-25" :model-type "cloud"}
   {:platform "Anthropic" :name "Claude 3.7 Sonnet" :model-ref "claude-3-7-sonnet-20250219" :price-in 3.0 :price-out 15.0 :parameters "?" :model-type "cloud"}
   {:platform "Anthropic" :name "Claude 3.5 Haiku" :model-ref "claude-3-5-haiku-20241022" :price-in 0.8 :price-out 4.0 :parameters "?" :model-type "cloud"}
   {:platform "Anthropic" :name "Claude 3 Haiku" :model-ref "claude-3-haiku-20240307" :price-in 0.25 :price-out 1.25 :parameters "?" :model-type "cloud"}])

;; A wrapper function to check which api to use
;; TODO: add in system prompt as option
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
