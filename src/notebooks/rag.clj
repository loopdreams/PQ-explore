;; # RAG Test
(ns notebooks.rag
  (:require [notebooks.question-vdb :refer [query-db-store add-question-to-store! embedding-model db-store]]
            [notebooks.preparation :refer [ds]]
            [clj-http.client :as client]
            [scicloj.kindly.v4.kind :as kind]
            [tablecloth.api :as tc]
            [scicloj.tableplot.v1.plotly :as plotly]
            [clojure.string :as str]
            [jsonista.core :as json]
            [wkok.openai-clojure.api :as api]
            [clojure.edn :as edn]
            [clojure.java.io :as io])
  (:import (dev.langchain4j.data.embedding Embedding)
           (dev.langchain4j.data.segment TextSegment)
           (dev.langchain4j.model.embedding EmbeddingModel)
           (dev.langchain4j.store.embedding EmbeddingMatch)
           (dev.langchain4j.store.embedding.inmemory InMemoryEmbeddingStore)
           (dev.langchain4j.model.embedding.onnx.allminilml6v2 AllMiniLmL6V2EmbeddingModel)))

;; A short test for demonstrating how to provide a LLM with context from existing similar questions.
;;

;; ## Setting up API calls

;; ### Building a prompt
;;
;; The functions below:
;;
;; 1. Take a question and lookup the 'question vector database' to find similar quesitons that have already been asked.
;;
;; 2. The answers to these previously asked questions are then given to the LLM as context for generating an answer.

(defn build-context [question]
  (let [similar-questions (map :text (query-db-store question 5))
        previous-answers (-> ds
                             (tc/select-rows #(some #{(% :question)} similar-questions))
                             :answer)]
    (str/join "\n" previous-answers)))

;; TODO: add in questions to context
(defn make-context-prompt [question]
  (let [ctx (build-context question)]
    (str "You are a responsible government official. Provide an informative, short answer to the user's question, using the supplied context only. Don't use any markup in your answer."
         " Context: " ctx)))

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
   {:platform "Ollama" :name "Llama3.1" :parameters "3B" :model-ref "llama3.2" :model-type "local"}
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



(kind/table (sort-by :name llm-models))

;; ## Running the Models and Storing Responses
;;
;; To save time, especially since the local models take a LONG time to run on my Macbook M1, responses
;; are written to the 'data' folder in edn format. The functions below help with this.

(defn unix-timestamp []
  (str (quot (System/currentTimeMillis) 1000)))

;; TODO: store responses for a question in folder, then, add all respsones later
(defn run-models-and-write-answers-to-disk! [models question directory & skip-context?]
  (let [save-location (str "data/" directory "/" (unix-timestamp) "-model_responses.edn")]
    (io/make-parents save-location)
    (spit save-location
          (reduce (fn [res {:keys [model-ref] :as model}]
                    (let [answer (ask-llm {:question question
                                           :model-ref model-ref
                                           :system-prompt (when-not skip-context? (make-context-prompt question))})]
                      (conj res (-> model
                                    (assoc :question question)
                                    (assoc :response answer)
                                    (assoc :context? (if skip-context? "without-context" "with-context"))))))
                  []
                  models))))

;; ### Test Questions
;;
;; Ideally, we would test the models on a large set of questions. For example, we could ask the model to
;; generate sample questions based on the material within the vector database.
;;
;; However, for this exercise, we will just look at some specific sample questions, to try to better understand
;; the responses that are produced.
;;
;; We will look at:
;;
;; 1. A real question that was asked
;;
;; 2. A question based on specific data in the dataset
;;
;; 3. A general/broad question
;;
;; #### A Real Question
;; Our dataset ranges from around January to March 2024, so we will take a
;; question that was asked after that time. You can see the question and its
;; answer [on the Oireachtas website](https://www.oireachtas.ie/en/debates/question/2024-04-09/972/)
;;
;; I've added the word 'Irish' to the question for the times when we are asking the model without providing context.

(def sample-question-1 "Deputy Michael Lowry asked the Irish Minister for Agriculture, Food and the Marine if conversion of conifer crops to native woodland is permitted under the Native Woodland Conservation scheme 2023-2027 recently launched by his Department; and if he will make a statement on the matter.")


;; #### A Targeted Question
;; To find what kind of information we'd like to target, let's look at some answers that contain figures

(-> ds
    (tc/drop-missing :answer)
    (tc/select-rows #(re-find #"\d+" (% :answer)))
    (tc/select-rows (range 10 11))
    :answer)

;; The answer here, about the 'Microgeneration Support Scheme', will do.

(def sample-question-2 "When was the Microgeneration Support Scheme approved by the Irish Government, and what is the maximum amount available for a grant?")

;; The expected answers for this question would be
;;
;; - 21 December 2021
;;
;; - 2,100 EUR


;; #### A general question

(def sample-question-3 "What is the Irish government doing to tackle climate change?")

;;
(def test-question-1 "Deputy Holly Cairns asked the Irish Minister for Education if SNA training will be reviewed in order that SNAs receive a level of training before they enter the classroom.")
#_(def test-question-2 "Deputy Paul McAuliffe asked the Minister for Education if it is planned to provide further supports to Irish schools experiencing difficulties with funding and increased costs of utilities; and if she will make a statement on the matter.")
(def test-question-2 "Deputy Peter 'Chap' Cleere asked the Irish Minister for Enterprise, Trade and Employment if he will report on the south east regional enterprise plan; if a new plan is being prepared; and if he will make a statement on the matter.")
(def test-question-3 "Deputy Emer Currie asked the Minister for Transport his plans to expand EV charging points at the State's airports to facilitate more EV drivers and an increase in EV car rental; and if he will make a statement on the matter.")

;; ### Running the models:
(comment
  ;; test-run
  (-> (filterv #(= (:model-ref %) "gemini-2.0-flash") llm-models)
      (run-models-and-write-answers-to-disk! sample-question-1 "testing_question_1"))
  (-> (filterv #(= (:model-ref %) "gpt-3.5-turbo") llm-models)
      (run-models-and-write-answers-to-disk! sample-question-1 "testing_question_1"))






  (time (run-models-and-write-answers-to-disk! local-llm-models test-question-1 ask-llm-local "question_1"))
  (time (run-models-and-write-answers-to-disk! openai-llm-models test-question-1 ask-llm-openai "question_1"))
  (time (run-models-and-write-answers-to-disk! google-llm-models test-question-1 ask-llm-google "question_1"))
  (time (run-models-and-write-answers-to-disk! anthropic-llm-models test-question-1 ask-llm-claude "question_1"))

  ;; Question 2
  (time (run-models-and-write-answers-to-disk! local-llm-models test-question-2 ask-llm-local "question_2"))
  ;; "Elapsed time: 418235.314792 msecs"
  (time (run-models-and-write-answers-to-disk! openai-llm-models test-question-2 ask-llm-openai "question_2"))
  ;; "Elapsed time: 5165.15375 msecs"
  (time (run-models-and-write-answers-to-disk! google-llm-models test-question-2 ask-llm-google "question_2"))
  ;; "Elapsed time: 12347.751916 msecs"
  (time (run-models-and-write-answers-to-disk! anthropic-llm-models test-question-2 ask-llm-claude "question_2"))
  ;; "Elapsed time: 15730.635083 msecs"

  ;; Question 2 - no context provided
  (time (run-models-and-write-answers-to-disk! local-llm-models test-question-2 ask-llm-local "question_2" true))
  ;; "Elapsed time: 375947.230834 msecs"
  (time (run-models-and-write-answers-to-disk! openai-llm-models test-question-2 ask-llm-openai "question_2" true))
  ;; "Elapsed time: 5786.090792 msecs"
  (time (run-models-and-write-answers-to-disk! google-llm-models test-question-2 ask-llm-google "question_2" true))
  ;; "Elapsed time: 25428.931375 msecs"
  ;; "Elapsed time: 14198.579708 msecs" - below
  (time (run-models-and-write-answers-to-disk! anthropic-llm-models test-question-2 ask-llm-claude "question_2" true))

  ;; Question 3
  (time (run-models-and-write-answers-to-disk! local-llm-models test-question-3 ask-llm-local "question_3"))
  (time (run-models-and-write-answers-to-disk! openai-llm-models test-question-3 ask-llm-openai "question_3"))
  (time (run-models-and-write-answers-to-disk! google-llm-models test-question-3 ask-llm-google "question_3"))
  (time (run-models-and-write-answers-to-disk! anthropic-llm-models test-question-3 ask-llm-claude "question_3"))
  ;; Question 3 - no context provided
  (time (run-models-and-write-answers-to-disk! local-llm-models test-question-3 ask-llm-local "question_3" true))
  (time (run-models-and-write-answers-to-disk! openai-llm-models test-question-3 ask-llm-openai "question_3" true))
  (time (run-models-and-write-answers-to-disk! google-llm-models test-question-3 ask-llm-google "question_3" true))
  (time (run-models-and-write-answers-to-disk! anthropic-llm-models test-question-3 ask-llm-claude "question_3" true)))


;; ### Combining Responses
;; I ran each of the types of models separately, mainly for debugging purposes. However, these could
;; also be run as a single batch. Since they are saved as separate batehes, this small function just
;; pulls the files together into a dataset to work with later.

(defn combine-model-responses [dir]
  (let [files (rest (file-seq (io/file dir)))]
    (reduce (fn [res file] (into res (edn/read-string (slurp file)))) [] files)))

(def model-responses-question-1 (combine-model-responses "data/question_1"))

(def testing-model-responses-question-1 (combine-model-responses "data/testing_question_1"))

(kind/table
 testing-model-responses-question-1)

(comment
  (def model-responses-question-2 (combine-model-responses "data/question_2"))
  (def model-responses-question-test (combine-model-responses "data/test_question_1")))
;; A quick look at what is inside these reponses,.
(kind/pprint (first model-responses-question-1))


;; ## Testing Responses
;; To test the responses, we will firstly build a vector database of all previous answers. This
;; is the same process that was used to build the questions vector database.
;;
;; This approach is quite crude, but I am exploring it here as a starting point.
;; In the future, it might be better to explore other metrics that are used in
;; this space such as:
;;
;; - ROUGE
;; - Token Overlap
;; - BLEU
;; - DeBERT (for measureing semantic equivalence)
;;

;; TODO: These are stored as file to avoid always loading them. For final version uncomment these.
(comment
  (def db-store-answers-temp (InMemoryEmbeddingStore/new))
  (def answers-list (-> ds (tc/drop-missing :answer) :answer))
  (count (map #(add-question-to-store! (str %) db-store-answers-temp) answers-list))
  (spit "data/db-store-answers.json" (.serializeToJson db-store-answers)))


(def db-store-answers (InMemoryEmbeddingStore/fromFile "data/db-store-answers.json"))

;; ### Similarity 'Score'
;; The measurement method is fairly crude, but this reflects the data we are dealing with - unstructured text. My approach here is to
;; simply retrieve the 5 most similar answers from the vector database, and record the min, max and average of these 5 scores.
(defn add-similarity-scores [model-data store]
  (let [query  (.content (. embedding-model embed (:response model-data)))
        result (. store findRelevant query 5)
        scores (mapv #(.score %) result)
        max    (first scores)
        min    (last scores)
        avg    (float (/ (apply + scores) (count scores)))]
    (-> model-data
        (assoc :max max)
        (assoc :min min)
        (assoc :avg avg)
        (assoc :scores scores))))

;; Let's see what the data looks like after adding the scores:

(-> (first model-responses-question-1)
    (add-similarity-scores db-store-answers)
    (kind/pprint))

;; Adding the scores to all the data previously generated
(def model-scores-q1 (mapv #(add-similarity-scores % db-store-answers) model-responses-question-1))
(def model-scores-testing (mapv #(add-similarity-scores % db-store-answers) testing-model-responses-question-1))
(comment
  (def model-scores-q2 (mapv #(add-similarity-scores % db-store-answers) model-responses-question-2))
  (def model-scores-q-test (mapv #(add-similarity-scores % db-store-answers) model-responses-question-test)))

;; To test that the vector database is actually working as intended, I have added a 'nonsense' response here, which
;; should not be similar at all to the responses in the database. The text is the first two verses from Lewis Carrol's
;; poem "The Hunting of the Snark"

(def nonsense-response
  {:model-ref "control"
   :response "\"Just the place for a Snark!\" the Bellman cried, As he landed his crew with care; Supporting each man on the top of the tide By a finger entwined in his hair.\n \"Just the place for a Snark! I have said it twice: That alone should encourage the crew. Just the place for a Snark! I have said it thrice: What I tell you three times is true.\""
   :context? "nonsense"})

(def nonsense-response-scores (add-similarity-scores nonsense-response db-store-answers))


;; #### Visualising LLM Performance

;; This function just pivots the previous data so that it can be used for box plots.
(defn box-plot-data-layout [data]
  (reduce (fn [res {:keys [scores] :as m-data}]
            (into res (map #(assoc m-data :score %) scores)))
          [] data))


(-> (sort-by :avg model-scores-testing)
    (box-plot-data-layout)
    (tc/dataset)
    (tc/order-by :max :desc)
    (plotly/base {:=width 800 :=height 400})
    (plotly/layer-boxplot
     {:=x :model-ref
      :=y :score
      :=color :context?}))


;; Another box plot, using Vega Lite. This time we will also include the 'control'
(kind/vega-lite
 {:data {:values (->> (reverse (sort-by :max (box-plot-data-layout (concat model-scores-testing [nonsense-response-scores]))))
                      (mapv (fn [m-data]
                              (update m-data :model-ref (partial str (str (:context? m-data) "-"))))))}
  :width 600
  :mark {:type "boxplot"
         :extent "min-max"}
  :encoding {:y {:field :model-ref :type :nominal
                 :sort false}
             :x {:field :score :type :quantitative
                 :scale {:zero false}}
             :color {:field :context? :type :nominal}}})

;; Finally let's join all the question responses together and plot the best models

(-> (sort-by :avg (concat model-scores-q1 [nonsense-response-scores]))
    (box-plot-data-layout)
    (tc/dataset)
    (plotly/base {:=width 800 :=height 400})
    (plotly/layer-boxplot
     {:=x :model-ref
      :=y :score
      :=color :context?}))





;; ### LLM 'Supervisor'
;;
;; In the above comparisons, gemini 2 Flash seems to do quite well with this type of exercise. Let's try to insert that model as a
;; 'supervisor' for the other models, and see if it can help improve the answers.

(defn ask-llm-supervisor-to-improve [{:keys [question response]}]
  (-> (client/post (str "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key="
                        (:gemini-api-key (edn/read-string (slurp "secrets.edn"))))
                   {:form-params
                    {:contents
                     {:parts
                      [{:text (str "You are a dilligent senior official at a government department. Try to improve the following answer, based on the context provided. Try to keep the new answer as short as possible."
                                   "QUESTION: " question
                                   "ANSWER: " response
                                   "CONTEXT: " (build-context question))}]}}
                    :content-type :json})
      :body
      (json/read-value json/keyword-keys-object-mapper)
      :candidates first :content :parts first :text))

(defn format-model-responses [m-data-all]
  (->> (mapv :response m-data-all)
       (map-indexed vector)
       (mapv (fn [[idx resp]]
               (str "--Beginning of Reponse Number " (str (inc idx)) " --"
                    resp
                    "--End of Response Number " (str (inc idx)) " --")))
       (str/join "\n")))

(defn ask-llm-supervisor-ranking [m-data-all supervisor-model]
  (let [model-responses (format-model-responses m-data-all)
        question (:question (first m-data-all))
        context (build-context question)
        query (str/join "\n"
                        ["You are a dilligent senior official at a government department. Based on the following question and context, rank the subsequent responses in order of best to worst. Please also include a simple json-friendly list of the ranking numbers at the end."
                         (str "Question: " question)
                         (str "Context: " context "--- End of Context ---")
                         model-responses])]
    (ask-llm {:model-ref supervisor-model :question query})))

(comment
  (spit "data/testing_google_supervisor.txt" (ask-llm-supervisor-ranking model-scores-testing "gemini-2.0-flash")))

(comment
  (defonce test-ranking
    (ask-llm-supervisor-ranking model-responses-question-1))

  (defonce test-ranking-2
    (ask-llm-supervisor-ranking model-responses-question-1))
  (defonce test-ranking-3
    (ask-llm-supervisor-ranking model-responses-question-1))
  (defonce test-ranking-GPT
    (ask-llm-supervisor-ranking-gpt model-responses-question-1)))

(def test-ranking-results [10 19 32 16 15 30 31 13 9 14 20 12 11 29 21 17 24 25 18 27 1 3 2 28 23 7 4 22 5 26 8])
(def test-ranking-results-re-run [10 3 9 27 28 21 12 14 6 13 17 11 29 18 16 15 19 32 30 31 24 25 2 1 23 4 20 7 26 8 22 5])
(def test-ranking-results-GPT [21 18 6 3 10 1 12 2 4 19 15 20 7 5 11 25 17 22 26 9])




(defn add-llm-ranking-score [m-data scores]
  (loop [[x & xs] scores
         idx (count scores)
         new-data []]
    (if-not x new-data
            (recur xs (dec idx)
                   (conj new-data (assoc  (nth m-data (dec x)) :llm-score idx))))))

(->
 (mapv #(add-similarity-scores % db-store-answers) (add-llm-ranking-score model-responses-question-1 test-ranking-results))
 (tc/dataset)
 (plotly/base
  {:=x :llm-score
   :=y :max})
 (plotly/layer-point)
 (plotly/layer-smooth))


(kind/md
 (:response
  (nth model-responses-question-1 9)))





(defn conduct-supervisor-revisions [m-data store]
  (let [supervisor-revision (ask-llm-supervisor-to-improve m-data)
        revision-score-query (.content (. embedding-model embed supervisor-revision))
        result (. store findRelevant revision-score-query 5)
        scores (mapv #(.score %) result)
        max (first scores)
        min (last scores)
        avg (float (/ (apply + scores) (count scores)))]
    (-> m-data
        (assoc :max-revised max)
        (assoc :min-revised min)
        (assoc :avg-revised avg)
        (assoc :scores-revised scores)
        (assoc :response-revised supervisor-revision))))

(comment
  (defonce revisions-added-q3 (mapv #(conduct-supervisor-revisions % db-store-answers)
                                    model-responses-question-1)))

(defonce revisions-added-q1 (edn/read-string (slurp "data/test_supervisor_revisions.edn")))



(-> (mapv #(add-similarity-scores % db-store-answers) revisions-added-q1)
    (tc/dataset)
    (tc/drop-rows #(= (:context? %) "without-context"))
    (tc/order-by :max :desc)
    (plotly/base
     {:=x :model-ref
      :=width 800
      :=y-title "% similar (max score)"
      :showlegend false})
    (plotly/layer-line
     {:=y :max
      :=mark-size 5})
    (plotly/layer-point
     {:=y :max
      :=mark-size 5})
    (plotly/layer-line
     {:=y :max-revised})
    (plotly/layer-point
     {:=y :max-revised}))









;; ### Sentence by Sentence Similarity

(defn similarity-test [text store]
  (if (= (str/trim text) "") 0
      (let [query (.content (. embedding-model embed text))
            result (. store findRelevant query 5)
            scores (mapv (fn [r] (.score r)) result)]
        (float (/ (reduce + scores) (count scores))))))

(defn split-sentences [text] (remove #(= % "") (str/split text #"(?<=\.) |\n")))

(defn response-sentence-scores [{:keys [response model-ref context?]}]
  (let [sentences (split-sentences response)
        data (mapv (fn [s] (let [score (similarity-test s db-store-answers)]
                             {:score score
                              :text s
                              :model-ref model-ref
                              :context? context?}))
                   sentences)]
    data))

(kind/table
 (response-sentence-scores (first model-scores-q1)))


(defn get-score-color [score]
  (condp > score
    0.65 "#F195A9" ;; red
    0.75 "#EAD196" ;; yellow
    0.85 "#E1EACD" ;; light green
    "#BAD8B6"))



;; TODO: preserve formatting (newlines, bullets, etc)
(defn color-response-score [sentence-scores]
  [:div
   [:h4 (str (:model-ref (first sentence-scores))
             " "
             (:context? (first sentence-scores)))]
   (into [:p]
         (map (fn [{:keys [score text]}]
                [:span {:style (str "background-color: " (get-score-color score))} (str text " ")])
              sentence-scores))])

;; Coloring response scores, from 'worst' to 'best'

(kind/md test-question-1)

(->> (sort-by :max (concat model-scores-q1 [nonsense-response-scores]))
     (mapv response-sentence-scores)
     (mapv color-response-score)
     (into [:div])
     (kind/hiccup))
